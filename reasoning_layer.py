"""
文档级关系抽取的语义推理模块。

本模块实现了"实体引导的证据累积"（IER）架构：

核心创新：
    - 固定3个语义槽位：[头实体查询, 尾实体查询, 桥接查询]
    - 使用实体嵌入作为"锚点"而非随机初始化
    - 可学习的桥接令牌通过推理捕获实体关系
    - 无模式坍塌风险（无随机槽位，无需正交性损失）

架构：
    1. 语义槽位：头/尾锚点 + 可学习桥接令牌
    2. 交叉注意力：槽位关注文档以检索证据
    3. 槽位间通信：自注意力用于信息交换
    4. KV预计算：优化高效处理

作者：重构为IER架构
性能：通过KV预计算进行优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, NamedTuple
import math


class PrecomputedKV(NamedTuple):
    """预计算的键值张量容器。"""
    keys: torch.Tensor      # [B, L, H]
    values: torch.Tensor    # [B, L, H]


class ReasoningLayer(nn.Module):
    """
    带有文档交叉注意力的单层推理层。
    
    性能优化：
        - 支持预计算的K/V以避免冗余投影
        - 使用高效的多头注意力实现
    
    架构：
        1. 交叉注意力：查询关注文档令牌
        2. FFN + LayerNorm：标准Transformer组件
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_ratio: int = 4
    ):
        """
        参数：
            hidden_size: 隐藏状态维度（例如，BERT-base为768）
            num_heads: 注意力头数
            dropout: Dropout概率
            ffn_ratio: FFN扩展比率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_p = dropout
        
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) 必须能被 num_heads ({num_heads}) 整除"
        
        # ===== 交叉注意力投影（显式用于KV预计算）=====
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.cross_norm = nn.LayerNorm(hidden_size)
        
        # ===== 前馈神经网络 =====
        ffn_hidden = hidden_size * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        
        # 残差连接的Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 注意力缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def precompute_kv(self, document: torch.Tensor) -> PrecomputedKV:
        """
        预计算文档的键和值投影。
        
        每个文档批次调用一次，然后在所有实体对块中重用。
        
        参数：
            document: 文档序列 [B, L, H]
        
        返回：
            包含键和值的PrecomputedKV，每个形状为 [B, L, H]
        """
        keys = self.k_proj(document)    # [B, L, H]
        values = self.v_proj(document)  # [B, L, H]
        return PrecomputedKV(keys=keys, values=values)
    
    def forward(
        self,
        query: torch.Tensor,
        document: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        precomputed_kv: Optional[PrecomputedKV] = None
    ) -> torch.Tensor:
        """
        单层推理的前向传播。
        
        参数：
            query: 槽位查询 [B, 3, H]（头、尾、桥接）
            document: 文档序列 [B, L, H]（如果提供了precomputed_kv则忽略）
            key_padding_mask: 填充掩码 [B, L]，True = 填充（忽略）
            precomputed_kv: 预计算的K/V（可选，用于提高效率）
        
        返回：
            evolved_query: 更新后的查询 [B, 3, H]
        """
        batch_size, num_slots, _ = query.shape
        seq_len = document.size(1) if precomputed_kv is None else precomputed_kv.keys.size(1)
        
        # ===== 步骤1：获取Q、K、V =====
        # 查询投影（始终重新计算）
        Q = self.q_proj(query)  # [B, 3, H]
        
        # 键/值：如果可用则使用预计算的
        if precomputed_kv is not None:
            K = precomputed_kv.keys    # [B, L, H]
            V = precomputed_kv.values  # [B, L, H]
        else:
            K = self.k_proj(document)  # [B, L, H]
            V = self.v_proj(document)  # [B, L, H]
        
        # ===== 步骤2：重塑为多头注意力 =====
        # [B, S, H] -> [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim]
        Q = Q.view(batch_size, num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Q: [B, num_heads, 3, head_dim]
        # K, V: [B, num_heads, L, head_dim]
        
        # ===== 步骤3：计算注意力 =====
        # 准备注意力掩码
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, L]，True = 忽略
            # 需要扩展以进行广播：[B, 1, 1, L]
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_mask = attn_mask.expand(batch_size, self.num_heads, num_slots, seq_len)
            attn_mask = attn_mask.to(dtype=Q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), float('-inf'))
        
        # 计算注意力分数
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, heads, 3, L]
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_dropped = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        
        attn_output = torch.matmul(attn_weights_dropped, V)  # [B, heads, 3, head_dim]
        
        # ===== 步骤4：重塑回并投影 =====
        # [B, num_heads, 3, head_dim] -> [B, 3, num_heads, head_dim] -> [B, 3, H]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_slots, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        # 残差 + LayerNorm
        query = self.cross_norm(query + self.dropout(attn_output))
        
        # ===== 步骤5：前馈神经网络 =====
        ffn_output = self.ffn(query)
        query = self.ffn_norm(query + ffn_output)
        
        return query


class InterSlotCommunication(nn.Module):
    """
    语义槽位间的自注意力（头/尾/桥接）。
    
    目的：
        允许桥接令牌聚合来自两个实体上下文的信息，
        实现跨实体推理。头和尾也可以交换信息。
    
    机制：
        在交叉注意力层之间对3个槽位应用标准自注意力。
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 输入：[B, 3, H]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        在槽位间应用自注意力。
        
        参数：
            slots: 语义槽位表示 [B, 3, H]
        
        返回：
            经过槽位间信息交换更新的槽位 [B, 3, H]
        """
        # 自注意力：槽位相互关注
        attn_output, _ = self.self_attn(
            query=slots,
            key=slots,
            value=slots,
            need_weights=False
        )
        
        # 残差 + LayerNorm
        return self.norm(slots + self.dropout(attn_output))


class GatedBridgeProjector(nn.Module):
    """
    使用门控交叉注意力的增强桥接查询生成器。
    
    解决审稿人关注的问题："桥接的线性瓶颈"
    
    而不是使用可能无法捕获多跳推理路径的简单MLP，
    该模块使用交叉注意力允许头和尾实体
    "查询"彼此的视角，然后门控融合。
    
    架构：
        1. 交叉注意力：H查询T的视角，T查询H的视角
        2. 门控融合：可学习的门控控制信息混合
        3. 残差 + LayerNorm：稳定训练
    
    数学直觉：
        对于关系 A -> B -> C，如果我们只有A和C的嵌入，
        标准的 MLP(concat(A, C)) 可能不会激活中间节点B。
        交叉注意力允许A"询问"C关于它们关系的信息，
        可能揭示通过B的路径信息。
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # H->T 和 T->H 视角的交叉注意力
        self.cross_attn_h2t = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_t2h = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 门控融合机制
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # 带残差路径的输出投影
        self.out_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        从头尾实体嵌入生成桥接查询。
        
        参数：
            h_emb: 头实体嵌入 [B, H]
            t_emb: 尾实体嵌入 [B, H]
        
        返回：
            bridge_query: 桥接查询嵌入 [B, H]
        """
        # 为注意力添加序列维度：[B, H] -> [B, 1, H]
        h_seq = h_emb.unsqueeze(1)
        t_seq = t_emb.unsqueeze(1)
        
        # 交叉注意力：H查询T的视角
        # "T知道什么与我(H)相关的信息？"
        h_to_t, _ = self.cross_attn_h2t(
            query=h_seq, key=t_seq, value=t_seq, need_weights=False
        )  # [B, 1, H]
        
        # 交叉注意力：T查询H的视角
        # "H知道什么与我(T)相关的信息？"
        t_to_h, _ = self.cross_attn_t2h(
            query=t_seq, key=h_seq, value=h_seq, need_weights=False
        )  # [B, 1, H]
        
        # 移除序列维度
        h_to_t = h_to_t.squeeze(1)  # [B, H]
        t_to_h = t_to_h.squeeze(1)  # [B, H]
        
        # 门控融合：学习如何混合双向视角
        combined = torch.cat([h_to_t, t_to_h], dim=-1)  # [B, 2H]
        gate = self.gate_proj(combined)  # [B, H]，值在[0, 1]范围内
        
        # 加权组合 + 线性投影
        fused = gate * h_to_t + (1 - gate) * t_to_h  # [B, H]
        
        # 最终投影带简单平均的残差
        residual = (h_emb + t_emb) / 2  # 简单基线
        output = self.out_proj(combined)  # [B, H]
        output = self.layernorm(output + self.dropout(fused) + residual * 0.1)
        
        return output


class DynamicAnchorUpdater(nn.Module):
    """
    动态锚点演化模块。
    
    解决审稿人关注的问题："范式风险 - 长距离指代消解能力退化"
    
    问题：
        如果头实体只出现在句子1中，但证据在句子10中
        使用代词"他"，则静态锚点（基于句子1）可能与
        句子10的上下文相似度较低（由于BERT的位置敏感表示）。
    
    解决方案：
        在每个推理层之后，通过融合原始
        实体嵌入和检索到的上下文来更新锚点。这允许锚点
        从文档中"吸收"共指信息。
    
    架构：
        1. 门控：学习更新多少（保留实体身份 vs 吸收上下文）
        2. 融合：原始和检索的加权组合
        3. LayerNorm：稳定表示
    
    信息瓶颈视角：
        门控作为信息瓶颈 - 它必须学会保留
        实体识别信息同时吸收与证据相关的上下文。
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 门控投影：决定更新多少
        # 输入：[原始锚点, 检索到的上下文]
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # 融合前转换检索到的上下文
        self.context_transform = nn.Linear(hidden_size, hidden_size)
        
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        original_anchor: torch.Tensor, 
        retrieved_context: torch.Tensor
    ) -> torch.Tensor:
        """
        通过与检索到的上下文融合来更新锚点。
        
        参数：
            original_anchor: 原始实体嵌入 [B, H]
            retrieved_context: 该锚点检索到的上下文 [B, H]
        
        返回：
            updated_anchor: 更新后的锚点嵌入 [B, H]
        """
        # 转换检索到的上下文
        transformed_ctx = self.context_transform(retrieved_context)
        
        # 计算更新门控
        combined = torch.cat([original_anchor, transformed_ctx], dim=-1)
        gate = self.gate_proj(combined)  # [B, H]，值在[0, 1]范围内
        
        # 门控融合：高门控值 = 保留原始，低门控值 = 吸收上下文
        # 我们使用 (1 - gate) * context 使梯度流更直观
        updated = gate * original_anchor + (1 - gate) * transformed_ctx
        
        # 训练时应用dropout（信息瓶颈）
        if self.training:
            updated = self.dropout(updated)
        
        # LayerNorm用于稳定表示
        return self.layernorm(updated)


class SemanticReasoner(nn.Module):
    """
    实体引导的证据累积推理器。
    
    核心创新 - 实体条件槽位：
        所有槽位从第一层开始就具有语义基础：
        
        1. 头实体查询：从头实体嵌入初始化
           - 检索与主语实体相关的证据
        2. 尾实体查询：从尾实体嵌入初始化
           - 检索与宾语实体相关的证据
        3. 桥接查询：通过 MLP(cat(H,T)) 从两个实体派生
           - 从第1层就知道两个实体（无"盲桥接"）
           - 捕获特定于关系的查询
    
    为什么使用实体条件桥接：
        全局可学习的bridge_token对于同一文档中的所有实体对
        在第1层是相同的，浪费计算。通过派生
        bridge = MLP(cat(H, T))，我们确保从第一个交叉注意力层开始
        就具有实体对特定的注意力。
    
    性能优化：
        - precompute_all_kv()：一次性计算所有层的K/V
        - 在实体对块之间重用预计算的K/V
    
    架构：
        对于每一层：
            1. 交叉注意力：所有3个槽位关注文档（证据检索）
            2. 槽位间通信：槽位交换信息（自注意力）
            3. FFN：非线性变换
    """
    
    # 固定的语义槽位数量
    NUM_SLOTS = 3  # [头、尾、桥接]
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        ffn_ratio: int = 4,
        # ===== 消融实验配置标志 =====
        use_multihop_reasoning: bool = True,   # 控制：GatedBridgeProjector + InterSlotCommunication
        use_longrange_modeling: bool = True    # 控制：DynamicAnchorUpdater + 各向异性校正
    ):
        """
        参数：
            hidden_size: 隐藏状态维度
            num_heads: 注意力头数
            num_layers: 堆叠的推理层数
            dropout: Dropout概率
            ffn_ratio: FFN扩展比率
            use_multihop_reasoning: 如果为False，禁用GatedBridgeProjector和InterSlotCommunication
            use_longrange_modeling: 如果为False，禁用DynamicAnchorUpdater和各向异性校正
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # ===== 消融标志 =====
        self.use_multihop_reasoning = use_multihop_reasoning
        self.use_longrange_modeling = use_longrange_modeling
        
        # ===== [改进1] 增强的桥接投影器 =====
        # 解决审稿人关注的问题："桥接的线性瓶颈"
        # 使用门控交叉注意力代替简单MLP进行多跳推理
        if use_multihop_reasoning:
            self.bridge_projector = GatedBridgeProjector(
                hidden_size=hidden_size,
                num_heads=4,
                dropout=dropout
            )
        else:
            # 消融：简单MLP回退
            self.bridge_projector = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            )
        
        # ===== [改进2] 动态锚点更新器 =====
        # 解决审稿人关注的问题："长距离指代消解能力退化"
        # 在每个推理层后更新锚点以吸收共指信息
        if use_longrange_modeling:
            self.anchor_updater = DynamicAnchorUpdater(
                hidden_size=hidden_size,
                dropout=dropout
            )
        else:
            self.anchor_updater = None  # 消融：静态锚点
        
        # ===== [改进3] 各向异性校正 =====
        # 解决审稿人关注的问题："实体Embedding各向异性"
        # L2归一化锚点的可学习缩放因子
        if use_longrange_modeling:
            self.anchor_scale = nn.Parameter(torch.ones(1) * math.sqrt(hidden_size))
        else:
            self.anchor_scale = None  # 消融：不进行归一化
        
        # ===== 交叉注意力层（证据检索）=====
        self.cross_attn_layers = nn.ModuleList([
            ReasoningLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                ffn_ratio=ffn_ratio
            )
            for _ in range(num_layers)
        ])
        
        # ===== 槽位间通信层（槽位混合）=====
        # 在每个交叉注意力后应用，允许桥接吸收H/T信息
        if use_multihop_reasoning:
            self.slot_comm_layers = nn.ModuleList([
                InterSlotCommunication(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])
        else:
            self.slot_comm_layers = None  # 消融：无槽位间通信
    
    def precompute_all_kv(self, document: torch.Tensor) -> List[PrecomputedKV]:
        """
        预计算所有层的K/V。
        
        每个文档批次调用一次（在处理实体对块之前）。
        
        参数：
            document: 文档序列 [B, L, H]
        
        返回：
            PrecomputedKV列表，每层一个
        """
        return [layer.precompute_kv(document) for layer in self.cross_attn_layers]
    
    def forward(
        self,
        document: torch.Tensor,
        h_anchors: torch.Tensor,
        t_anchors: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        precomputed_kv_list: Optional[List[PrecomputedKV]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        在文档上执行语义推理。
        
        参数：
            document: 文档令牌嵌入 [B, L, H]
            h_anchors: 头实体嵌入 [B, H]（头查询的语义锚点）
            t_anchors: 尾实体嵌入 [B, H]（尾查询的语义锚点）
            attention_mask: 填充掩码 [B, L]（1=有效，0=填充）
            precomputed_kv_list: 所有层的预计算K/V（可选）
        
        返回：
            h_context: 推理后以头为中心的上下文 [B, H]
            t_context: 推理后以尾为中心的上下文 [B, H]
            bridge_context: 关系/路径上下文 [B, H]
        """
        batch_size = h_anchors.size(0)
        
        # ===== [改进3] 各向异性校正 =====
        # L2归一化 + 可学习缩放以解决表示退化问题
        # 参考：Ethayarajh, 2019 - "How Contextual are Contextualized Word Representations?"
        if self.use_longrange_modeling:
            h_anchors_norm = F.normalize(h_anchors, p=2, dim=-1) * self.anchor_scale
            t_anchors_norm = F.normalize(t_anchors, p=2, dim=-1) * self.anchor_scale
        else:
            # 消融：不进行归一化，使用原始锚点
            h_anchors_norm = h_anchors
            t_anchors_norm = t_anchors
        
        # 保留原始用于动态更新（演化锚点）
        h_evolving = h_anchors_norm
        t_evolving = t_anchors_norm
        
        # ===== 步骤1：构建实体条件语义查询 =====
        # [改进1] 使用GatedBridgeProjector进行多跳推理
        if self.use_multihop_reasoning:
            bridge_query_raw = self.bridge_projector(h_anchors_norm, t_anchors_norm)  # [B, H]
        else:
            # 消融：简单MLP接收拼接输入
            bridge_query_raw = self.bridge_projector(torch.cat([h_anchors_norm, t_anchors_norm], dim=-1))  # [B, H]
        
        # [锚点Dropout] 防止捷径学习/实体先验利用
        # 如果桥接有完整的实体信息，它可能会忽略文档而变成
        # 纯MLP分类器（关系分类而非文档级关系抽取）
        # 通过丢弃一些实体信号，我们强制它从文档中检索
        if self.training:
            bridge_query_raw = F.dropout(bridge_query_raw, p=0.3, training=True)
        
        bridge_query = bridge_query_raw.unsqueeze(1)  # [B, 1, H]
        
        # 将锚点堆叠为查询：[头、尾、桥接]
        queries = torch.cat([
            h_evolving.unsqueeze(1),  # [B, 1, H] - 头查询（归一化锚点）
            t_evolving.unsqueeze(1),  # [B, 1, H] - 尾查询（归一化锚点）
            bridge_query              # [B, 1, H] - 桥接查询（门控交叉注意力）
        ], dim=1)  # [B, 3, H]
        
        # 为注意力准备key_padding_mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # 反转：1->False，0->True
        
        # ===== 步骤2：带动态锚点更新的迭代推理循环 =====
        for layer_idx in range(self.num_layers):
            # 如果可用则获取预计算的KV
            precomputed_kv = None
            if precomputed_kv_list is not None:
                precomputed_kv = precomputed_kv_list[layer_idx]
            
            # 交叉注意力：从文档检索证据
            queries = self.cross_attn_layers[layer_idx](
                queries, document, key_padding_mask, precomputed_kv
            )
            
            # ===== [改进2] 动态锚点更新 =====
            # 用检索到的上下文更新头/尾锚点以解决
            # 长距离共指问题（零指代与代词失效问题）
            if self.use_longrange_modeling:
                h_retrieved = queries[:, 0, :]  # 头的检索上下文
                t_retrieved = queries[:, 1, :]  # 尾的检索上下文
                
                # 通过融合原始和检索到的来演化锚点
                h_evolving = self.anchor_updater(h_evolving, h_retrieved)
                t_evolving = self.anchor_updater(t_evolving, t_retrieved)
                
                # 用演化的锚点更新查询以用于下一层
                # 这允许锚点在文档中"跟踪"共指
                queries = torch.cat([
                    h_evolving.unsqueeze(1),
                    t_evolving.unsqueeze(1),
                    queries[:, 2:3, :]  # 保持桥接不变（它已经通过槽位间通信吸收了H/T）
                ], dim=1)
            # 否则：消融 - 静态锚点，不更新
            
            # 槽位间通信：允许桥接吸收H/T信息
            if self.use_multihop_reasoning:
                queries = self.slot_comm_layers[layer_idx](queries)
            # 否则：消融 - 无槽位间通信
        
        # ===== 步骤3：提取输出上下文 =====
        # queries: [B, 3, H] -> 分割为 [头、尾、桥接]
        h_context = queries[:, 0, :]      # [B, H] - 以头为中心的上下文（已演化）
        t_context = queries[:, 1, :]      # [B, H] - 以尾为中心的上下文（已演化）
        bridge_context = queries[:, 2, :] # [B, H] - 关系上下文
        
        return h_context, t_context, bridge_context


def test_semantic_reasoner():
    """测试新的SemanticReasoner架构。"""
    print("=" * 60)
    print("测试 SemanticReasoner（IER架构）")
    print("=" * 60)
    
    # 配置
    batch_size = 4
    seq_len = 128
    hidden_size = 768
    num_layers = 2
    
    # 创建模块
    reasoner = SemanticReasoner(
        hidden_size=hidden_size,
        num_heads=8,
        num_layers=num_layers,
        dropout=0.1
    )
    
    print(f"\n[配置] batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
    print(f"[配置] num_layers={num_layers}, num_slots=3（固定：头/尾/桥接）")
    
    # 创建输入
    document = torch.randn(batch_size, seq_len, hidden_size)
    h_anchors = torch.randn(batch_size, hidden_size)  # 头实体嵌入
    t_anchors = torch.randn(batch_size, hidden_size)  # 尾实体嵌入
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"\n[输入形状]")
    print(f"  document: {document.shape}")
    print(f"  h_anchors: {h_anchors.shape}")
    print(f"  t_anchors: {t_anchors.shape}")
    
    # 测试无预计算
    print(f"\n[测试1] 无KV预计算的前向传播...")
    h_ctx, t_ctx, bridge_ctx = reasoner(document, h_anchors, t_anchors, attention_mask)
    print(f"  ✓ h_context: {h_ctx.shape}")
    print(f"  ✓ t_context: {t_ctx.shape}")
    print(f"  ✓ bridge_context: {bridge_ctx.shape}")
    
    # 测试预计算
    print(f"\n[测试2] 带KV预计算的前向传播...")
    precomputed_kv = reasoner.precompute_all_kv(document)
    print(f"  ✓ 预计算的KV数量: {len(precomputed_kv)}")
    print(f"  ✓ KV形状: keys={precomputed_kv[0].keys.shape}, values={precomputed_kv[0].values.shape}")
    
    h_ctx2, t_ctx2, bridge_ctx2 = reasoner(document, h_anchors, t_anchors, attention_mask, precomputed_kv)
    print(f"  ✓ 使用预计算KV的输出: h={h_ctx2.shape}, t={t_ctx2.shape}, bridge={bridge_ctx2.shape}")
    
    # 测试梯度流
    print(f"\n[测试3] 梯度流测试...")
    loss = h_ctx.sum() + t_ctx.sum() + bridge_ctx.sum()
    loss.backward()
    print(f"  ✓ 梯度计算成功")
    # 检查投影器梯度（bridge_token不再存在，已替换为bridge_projector）
    first_param = next(reasoner.bridge_projector.parameters())
    print(f"  ✓ bridge_projector梯度存在: {first_param.grad is not None}")
    
    # 参数统计
    total_params = sum(p.numel() for p in reasoner.parameters())
    print(f"\n[模型统计]")
    print(f"  总参数量: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("SemanticReasoner测试通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    test_semantic_reasoner()
