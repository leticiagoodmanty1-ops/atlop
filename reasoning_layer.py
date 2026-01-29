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


class StructuredSlotCommunication(nn.Module):
    """
    结构化槽位通信模块（带传递性推理约束）。
    
    核心创新（论文包装点："Transitive Reasoning Constraint"）：
        - 强制 Head/Tail 必须通过 Bridge 交换信息
        - 显式建模传递性推理：Head ↔ Bridge ↔ Tail
        - 禁止 Head ↔ Tail 直接通信（必须经过中介）
    
    掩码矩阵设计：
        Query/Key  | Head | Tail | Bridge |
        -----------|------|------|--------|
        Head       |  ✓   |  ✗   |   ✓    |  Head能看自己和Bridge
        Tail       |  ✗   |  ✓   |   ✓    |  Tail能看自己和Bridge
        Bridge     |  ✓   |  ✓   |   ✓    |  Bridge能看所有（中心聚合者）
    
    理论意义：
        这强制模型必须通过 Bridge 槽位交换 Head 和 Tail 的信息，
        而非简单的全连接自注意力。这符合关系推断的语义逻辑：
        关系证据往往分布在连接两个实体的文本路径中。
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 输入：[B, 3, H]
        )
        
        # 注册结构化掩码（True = 被遮蔽/不可见）
        # 索引: 0=Head, 1=Tail, 2=Bridge
        # 使用additive mask: -inf表示不可见
        reasoning_mask_bool = torch.tensor([
            [False, True,  False],  # Head: 能看自己和Bridge，不能直接看Tail
            [True,  False, False],  # Tail: 能看自己和Bridge，不能直接看Head
            [False, False, False]   # Bridge: 能看所有（中心聚合者）
        ], dtype=torch.bool)
        
        # 转换为additive attention mask
        reasoning_mask_float = torch.zeros(3, 3)
        reasoning_mask_float.masked_fill_(reasoning_mask_bool, float('-inf'))
        self.register_buffer('reasoning_mask', reasoning_mask_float)
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        在槽位间应用结构化自注意力。
        
        参数：
            slots: 语义槽位表示 [B, 3, H]
        
        返回：
            经过结构化信息交换更新的槽位 [B, 3, H]
        """
        # 使用结构化掩码的自注意力
        # PyTorch MultiheadAttention 的 attn_mask 会自动广播到所有 batch 和 heads
        attn_output, _ = self.self_attn(
            query=slots,
            key=slots,
            value=slots,
            attn_mask=self.reasoning_mask,  # [3, 3] 自动广播
            need_weights=False
        )
        
        # 残差 + LayerNorm
        return self.norm(slots + self.dropout_layer(attn_output))


# 保留旧类名作为别名，确保向后兼容
InterSlotCommunication = StructuredSlotCommunication


class DocMediatedBridgeProjector(nn.Module):
    """
    文档中介的多跳桥接投影器。
    
    核心创新（解决审稿人关注的"桥接线性瓶颈"问题）：
        - 使用文档作为推理中介，而非实体直接交互
        - 头实体通过文档查询尾实体相关证据（h → doc → t）
        - 真正的多跳推理路径，利用文档中的桥接实体
    
    性能优化：
        - 合并两次注意力为一次批量计算（h_query 和 t_query 堆叠）
        - 减少注意力头数（默认2头，参数量更少）
        - 支持预计算的 K/V 复用
    
    论文包装点："Document-Mediated Multi-hop Bridging"
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # 生成面向对方的查询：h想从doc中找什么关于t的信息
        self.h_query_gen = nn.Linear(hidden_size, hidden_size)
        self.t_query_gen = nn.Linear(hidden_size, hidden_size)
        
        # K/V 投影（用于预计算）
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj_attn = nn.Linear(hidden_size, hidden_size)
        
        # 关系路由门控：决定头/尾视角的相对重要性
        self.route_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # 输出投影 + 归一化
        self.out_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化门控为平衡状态（0.5）以稳定训练
        self._init_gate_balanced()
    
    def _init_gate_balanced(self):
        """初始化门控为平衡状态，防止训练初期不稳定。"""
        if hasattr(self.route_gate[-1], 'weight'):
            nn.init.zeros_(self.route_gate[-1].weight)
            nn.init.zeros_(self.route_gate[-1].bias)  # sigmoid(0) = 0.5
    
    def forward(
        self,
        h_emb: torch.Tensor,
        t_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        doc_mask: Optional[torch.Tensor] = None,
        precomputed_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        从头尾实体嵌入生成桥接查询，使用文档作为推理中介。
        
        参数：
            h_emb: 头实体嵌入 [B, H]
            t_emb: 尾实体嵌入 [B, H]
            doc_emb: 文档表示 [B, L, H] - 作为推理桥梁的 L 个 token
            doc_mask: 填充掩码 [B, L]，True = 填充（忽略）
            precomputed_kv: 可选的预计算 (K, V)，形状各为 [B, L, H]
        
        返回：
            bridge_query: 桥接查询嵌入 [B, H]
        """
        batch_size = h_emb.size(0)
        seq_len = doc_emb.size(1)
        
        # 生成查询并堆叠（合并两次注意力为一次）
        h_query = self.h_query_gen(h_emb)  # [B, H]
        t_query = self.t_query_gen(t_emb)  # [B, H]
        # 堆叠为 [B, 2, H]
        queries = torch.stack([h_query, t_query], dim=1)
        
        # K/V：使用预计算或现场计算
        if precomputed_kv is not None:
            K, V = precomputed_kv
        else:
            K = self.k_proj(doc_emb)  # [B, L, H]
            V = self.v_proj(doc_emb)  # [B, L, H]
        
        # 重塑为多头注意力格式
        Q = queries.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, 2, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, L, head_dim]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, L, head_dim]
        
        # 计算注意力分数
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, heads, 2, L]
        
        # 应用掩码
        if doc_mask is not None:
            attn_mask = doc_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_layer.p, training=self.training)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # [B, heads, 2, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 2, self.hidden_size)  # [B, 2, H]
        attn_output = self.out_proj_attn(attn_output)  # [B, 2, H]
        
        # 拆分为 h_ctx 和 t_ctx
        h_ctx = attn_output[:, 0, :]  # [B, H]
        t_ctx = attn_output[:, 1, :]  # [B, H]
        
        # 门控融合：学习哪些路径证据更重要
        gate = self.route_gate(torch.cat([h_emb, t_emb], dim=-1))  # [B, H]
        fused = gate * h_ctx + (1 - gate) * t_ctx  # [B, H]
        
        # 残差连接（原始实体信息）+ 融合证据
        residual = (h_emb + t_emb) / 2
        combined = torch.cat([fused, residual], dim=-1)  # [B, 2H]
        
        output = self.out_proj(combined)  # [B, H]
        output = self.norm(output + self.dropout_layer(fused))
        
        return output
    
    def precompute_kv(self, doc_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预计算 K/V 用于复用。"""
        K = self.k_proj(doc_emb)
        V = self.v_proj(doc_emb)
        return (K, V)


class BridgeProjector(nn.Module):
    """
    统一接口的桥接投影器包装器。
    
    目的：
        保持消融实验的接口兼容性。无论 use_doc_mediated 是 True 还是 False，
        调用方始终使用相同的接口：forward(h_emb, t_emb, doc_emb, doc_mask)
    
    性能优化：
        支持 K/V 预计算复用，避免重复计算文档投影
    
    消融设置：
        - use_doc_mediated=True:  使用 DocMediatedBridgeProjector（完整实现）
        - use_doc_mediated=False: 使用简单 MLP（忽略 doc_emb，回退基线）
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 2,  # 减少默认头数以提升性能
        dropout: float = 0.1,
        use_doc_mediated: bool = True
    ):
        super().__init__()
        self.use_doc_mediated = use_doc_mediated
        self.hidden_size = hidden_size
        
        if use_doc_mediated:
            self.core = DocMediatedBridgeProjector(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            # 消融回退：简单 MLP，忽略 doc_emb
            self.core = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
    
    def forward(
        self,
        h_emb: torch.Tensor,
        t_emb: torch.Tensor,
        doc_emb: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
        precomputed_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        统一接口：始终接受 doc_emb 和 doc_mask，内部决定是否使用。
        
        参数：
            h_emb: 头实体嵌入 [B, H]
            t_emb: 尾实体嵌入 [B, H]
            doc_emb: 文档表示 [B, L, H]（消融时忽略）
            doc_mask: 填充掩码 [B, L]（消融时忽略）
            precomputed_kv: 预计算的 (K, V)，用于性能优化
        
        返回：
            bridge_query: 桥接查询嵌入 [B, H]
        """
        if self.use_doc_mediated:
            return self.core(h_emb, t_emb, doc_emb, doc_mask, precomputed_kv)
        else:
            # 回退：不使用 doc_emb，保持接口统一
            return self.core(torch.cat([h_emb, t_emb], dim=-1))
    
    def precompute_kv(self, doc_emb: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """预计算 K/V 用于复用。消融模式下返回 None。"""
        if self.use_doc_mediated:
            return self.core.precompute_kv(doc_emb)
        return None


# 保留旧类名作为别名，确保向后兼容
GatedBridgeProjector = BridgeProjector


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
        # 使用文档中介的交叉注意力实现真正的多跳推理
        # 注意：使用 BridgeProjector 包装器确保接口统一
        self.bridge_projector = BridgeProjector(
            hidden_size=hidden_size,
            num_heads=4,
            dropout=dropout,
            use_doc_mediated=use_multihop_reasoning  # 消融时自动切换到 MLP
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
    
    def precompute_all_kv(self, document: torch.Tensor) -> Tuple[List[PrecomputedKV], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        预计算所有层的K/V以及桥接投影器的K/V。
        
        每个文档批次调用一次（在处理实体对块之前）。
        
        参数：
            document: 文档序列 [B, L, H]
        
        返回：
            (reasoning_layers_kv, bridge_kv):
                - reasoning_layers_kv: PrecomputedKV列表，每层一个
                - bridge_kv: 桥接投影器的预计算 (K, V)，消融模式下为 None
        """
        reasoning_kv = [layer.precompute_kv(document) for layer in self.cross_attn_layers]
        bridge_kv = self.bridge_projector.precompute_kv(document)
        return (reasoning_kv, bridge_kv)
    
    def forward(
        self,
        document: torch.Tensor,
        h_anchors: torch.Tensor,
        t_anchors: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        precomputed_kv_list: Optional[Tuple[List[PrecomputedKV], Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
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
        
        # ===== 预先准备掩码 =====
        # 重要：必须在 bridge_projector 调用之前定义
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # 反转：1->False，0->True（填充位置）
        
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
        # [改进1] 使用 DocMediatedBridgeProjector 进行真正的多跳推理
        # 通过文档作为中介：h -> doc -> t
        # 性能优化：使用预计算的 K/V 避免重复计算
        bridge_kv = None
        reasoning_kv_list = None
        if precomputed_kv_list is not None:
            reasoning_kv_list, bridge_kv = precomputed_kv_list
        
        bridge_query_raw = self.bridge_projector(
            h_anchors_norm, t_anchors_norm, 
            document, key_padding_mask,
            precomputed_kv=bridge_kv  # 使用预计算的 K/V
        )  # [B, H]
        
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
        
        # ===== 步骤2：带动态锚点更新的迭代推理循环 =====
        for layer_idx in range(self.num_layers):
            # 如果可用则获取预计算的KV
            precomputed_kv = None
            if reasoning_kv_list is not None:
                precomputed_kv = reasoning_kv_list[layer_idx]
            
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

