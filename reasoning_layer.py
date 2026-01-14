"""
Semantic Reasoning Module for Document-level Relation Extraction.

This module implements the "Entity-Guided Evidence Accumulation" (IER) architecture:

Core Innovation:
    - Fixed 3 semantic slots: [Head_Query, Tail_Query, Bridge_Query]
    - Use entity embeddings as "anchors" instead of random initialization
    - Learnable Bridge Token captures entity relationships through reasoning
    - NO mode collapse risk (no random slots, no orthogonality loss needed)

Architecture:
    1. Semantic Slots: Head/Tail anchors + learnable Bridge token
    2. Cross-Attention: Slots attend to document for evidence retrieval
    3. Inter-Slot Communication: Self-attention for information exchange
    4. KV Pre-computation: Optimized for efficient processing

Author: Refactored to IER architecture
Performance: Optimized with KV pre-computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, NamedTuple
import math


class PrecomputedKV(NamedTuple):
    """Container for pre-computed Key and Value tensors."""
    keys: torch.Tensor      # [B, L, H]
    values: torch.Tensor    # [B, L, H]


class ReasoningLayer(nn.Module):
    """
    Single reasoning layer with cross-attention to document.
    
    PERFORMANCE OPTIMIZATION:
        - Supports pre-computed K/V to avoid redundant projections
        - Uses efficient multi-head attention implementation
    
    Architecture:
        1. Cross-Attention: Queries attend to document tokens
        2. FFN + LayerNorm: Standard transformer components
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_ratio: int = 4
    ):
        """
        Args:
            hidden_size: Dimension of hidden states (e.g., 768 for BERT-base)
            num_heads: Number of attention heads
            dropout: Dropout probability
            ffn_ratio: FFN expansion ratio
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_p = dropout
        
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
        # ===== Cross-Attention Projections (Explicit for KV pre-computation) =====
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.cross_norm = nn.LayerNorm(hidden_size)
        
        # ===== Feed-Forward Network =====
        ffn_hidden = hidden_size * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def precompute_kv(self, document: torch.Tensor) -> PrecomputedKV:
        """
        Pre-compute Key and Value projections for document.
        
        Call this ONCE per document batch, then reuse for all entity pair chunks.
        
        Args:
            document: Document sequence [B, L, H]
        
        Returns:
            PrecomputedKV containing keys and values, each [B, L, H]
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
        Forward pass for single reasoning layer.
        
        Args:
            query: Slot queries [B, 3, H] (Head, Tail, Bridge)
            document: Document sequence [B, L, H] (ignored if precomputed_kv provided)
            key_padding_mask: Padding mask [B, L], True = padded (ignore)
            precomputed_kv: Pre-computed K/V (optional, for efficiency)
        
        Returns:
            evolved_query: Updated queries [B, 3, H]
        """
        batch_size, num_slots, _ = query.shape
        seq_len = document.size(1) if precomputed_kv is None else precomputed_kv.keys.size(1)
        
        # ===== Step 1: Get Q, K, V =====
        # Query projection (always computed fresh)
        Q = self.q_proj(query)  # [B, 3, H]
        
        # Key/Value: use precomputed if available
        if precomputed_kv is not None:
            K = precomputed_kv.keys    # [B, L, H]
            V = precomputed_kv.values  # [B, L, H]
        else:
            K = self.k_proj(document)  # [B, L, H]
            V = self.v_proj(document)  # [B, L, H]
        
        # ===== Step 2: Reshape for multi-head attention =====
        # [B, S, H] -> [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim]
        Q = Q.view(batch_size, num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Q: [B, num_heads, 3, head_dim]
        # K, V: [B, num_heads, L, head_dim]
        
        # ===== Step 3: Compute attention =====
        # Prepare attention mask
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, L], True = ignore
            # Need to expand for broadcasting: [B, 1, 1, L]
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_mask = attn_mask.expand(batch_size, self.num_heads, num_slots, seq_len)
            attn_mask = attn_mask.to(dtype=Q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), float('-inf'))
        
        # Compute attention scores
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, heads, 3, L]
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_dropped = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        
        attn_output = torch.matmul(attn_weights_dropped, V)  # [B, heads, 3, head_dim]
        
        # ===== Step 4: Reshape back and project =====
        # [B, num_heads, 3, head_dim] -> [B, 3, num_heads, head_dim] -> [B, 3, H]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_slots, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        # Residual + LayerNorm
        query = self.cross_norm(query + self.dropout(attn_output))
        
        # ===== Step 5: Feed-Forward Network =====
        ffn_output = self.ffn(query)
        query = self.ffn_norm(query + ffn_output)
        
        return query


class InterSlotCommunication(nn.Module):
    """
    Self-attention among semantic slots (Head/Tail/Bridge).
    
    Purpose:
        Allows the Bridge token to aggregate information from both entity contexts,
        enabling cross-entity reasoning. Head and Tail can also exchange information.
    
    Mechanism:
        Standard self-attention over the 3 slots, applied between cross-attention layers.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input: [B, 3, H]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention among slots.
        
        Args:
            slots: Semantic slot representations [B, 3, H]
        
        Returns:
            Updated slots with inter-slot information exchange [B, 3, H]
        """
        # Self-attention: slots attend to each other
        attn_output, _ = self.self_attn(
            query=slots,
            key=slots,
            value=slots,
            need_weights=False
        )
        
        # Residual + LayerNorm
        return self.norm(slots + self.dropout(attn_output))


class GatedBridgeProjector(nn.Module):
    """
    Enhanced Bridge Query Generator using Gated Cross-Attention.
    
    ADDRESSES REVIEWER CONCERN: "桥接的线性瓶颈"
    
    Instead of a simple MLP that may fail to capture multi-hop reasoning paths,
    this module uses cross-attention to allow Head and Tail entities to 
    "query" each other's perspective, then gates the fusion.
    
    Architecture:
        1. Cross-Attention: H queries T's perspective, T queries H's perspective
        2. Gated Fusion: Learnable gate controls information blend
        3. Residual + LayerNorm: Stable training
    
    Mathematical Intuition:
        For relation A -> B -> C, if we only have embeddings for A and C,
        standard MLP(concat(A, C)) may not activate intermediate node B.
        Cross-attention allows A to "ask" C what C knows about their relationship,
        potentially surfacing information about the path through B.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Cross-attention for H->T and T->H perspectives
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
        
        # Gated fusion mechanism
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Output projection with residual path
        self.out_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Generate bridge query from head and tail entity embeddings.
        
        Args:
            h_emb: Head entity embedding [B, H]
            t_emb: Tail entity embedding [B, H]
        
        Returns:
            bridge_query: Bridge query embedding [B, H]
        """
        # Add sequence dimension for attention: [B, H] -> [B, 1, H]
        h_seq = h_emb.unsqueeze(1)
        t_seq = t_emb.unsqueeze(1)
        
        # Cross-attention: H queries T's perspective
        # "What does T know that's relevant to me (H)?"
        h_to_t, _ = self.cross_attn_h2t(
            query=h_seq, key=t_seq, value=t_seq, need_weights=False
        )  # [B, 1, H]
        
        # Cross-attention: T queries H's perspective
        # "What does H know that's relevant to me (T)?"
        t_to_h, _ = self.cross_attn_t2h(
            query=t_seq, key=h_seq, value=h_seq, need_weights=False
        )  # [B, 1, H]
        
        # Remove sequence dimension
        h_to_t = h_to_t.squeeze(1)  # [B, H]
        t_to_h = t_to_h.squeeze(1)  # [B, H]
        
        # Gated fusion: learn how to blend bidirectional perspectives
        combined = torch.cat([h_to_t, t_to_h], dim=-1)  # [B, 2H]
        gate = self.gate_proj(combined)  # [B, H], values in [0, 1]
        
        # Weighted combination + linear projection
        fused = gate * h_to_t + (1 - gate) * t_to_h  # [B, H]
        
        # Final projection with residual from simple average
        residual = (h_emb + t_emb) / 2  # Simple baseline
        output = self.out_proj(combined)  # [B, H]
        output = self.layernorm(output + self.dropout(fused) + residual * 0.1)
        
        return output


class DynamicAnchorUpdater(nn.Module):
    """
    Dynamic Anchor Evolution Module.
    
    ADDRESSES REVIEWER CONCERN: "范式风险 - 长距离指代消解能力退化"
    
    Problem:
        If Head entity only appears in sentence 1 but evidence is in sentence 10
        using a pronoun "He", the static anchor (based on sentence 1) may have
        low similarity with the context at sentence 10 (due to BERT's position-
        sensitive representations).
    
    Solution:
        After each reasoning layer, update the anchor by fusing the original
        entity embedding with the retrieved context. This allows the anchor
        to "absorb" coreference information from the document.
    
    Architecture:
        1. Gate: Learn how much to update (preserve entity identity vs absorb context)
        2. Fusion: Weighted combination of original and retrieved
        3. LayerNorm: Stabilize representations
    
    Information Bottleneck Perspective:
        The gate acts as an information bottleneck - it must learn to preserve
        entity-identifying information while absorbing evidence-relevant context.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Gate projection: decides how much to update
        # Input: [original_anchor, retrieved_context]
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Transform the retrieved context before fusion
        self.context_transform = nn.Linear(hidden_size, hidden_size)
        
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        original_anchor: torch.Tensor, 
        retrieved_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Update anchor by fusing with retrieved context.
        
        Args:
            original_anchor: Original entity embedding [B, H]
            retrieved_context: Context retrieved by this anchor [B, H]
        
        Returns:
            updated_anchor: Updated anchor embedding [B, H]
        """
        # Transform retrieved context
        transformed_ctx = self.context_transform(retrieved_context)
        
        # Compute update gate
        combined = torch.cat([original_anchor, transformed_ctx], dim=-1)
        gate = self.gate_proj(combined)  # [B, H], values in [0, 1]
        
        # Gated fusion: high gate = preserve original, low gate = absorb context
        # We use (1 - gate) * context to make gradient flow intuitive
        updated = gate * original_anchor + (1 - gate) * transformed_ctx
        
        # Apply dropout during training (information bottleneck)
        if self.training:
            updated = self.dropout(updated)
        
        # LayerNorm for stable representations
        return self.layernorm(updated)


class SemanticReasoner(nn.Module):
    """
    Entity-Guided Evidence Accumulation Reasoner.
    
    CORE INNOVATION - Entity-Conditioned Slots:
        All slots are semantically grounded from the very first layer:
        
        1. Head_Query: Initialized from head entity embedding
           - Retrieves evidence related to the subject entity
        2. Tail_Query: Initialized from tail entity embedding
           - Retrieves evidence related to the object entity
        3. Bridge_Query: Derived from BOTH entities via MLP(cat(H,T))
           - Knows about both entities from layer 1 (no "blind bridge")
           - Captures relationship-specific queries
    
    WHY ENTITY-CONDITIONED BRIDGE:
        A global learnable bridge_token would be identical for all entity pairs
        in the same document at layer 1, wasting computation. By deriving
        bridge = MLP(cat(H, T)), we ensure entity-pair-specific attention from
        the very first cross-attention layer.
    
    PERFORMANCE OPTIMIZATION:
        - precompute_all_kv(): Compute K/V for all layers at once
        - Reuse pre-computed K/V across entity pair chunks
    
    Architecture:
        For each layer:
            1. Cross-Attention: All 3 slots attend to document (evidence retrieval)
            2. Inter-Slot Communication: Slots exchange information (self-attention)
            3. FFN: Non-linear transformation
    """
    
    # Fixed number of semantic slots
    NUM_SLOTS = 3  # [Head, Tail, Bridge]
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        ffn_ratio: int = 4
    ):
        """
        Args:
            hidden_size: Dimension of hidden states
            num_heads: Number of attention heads
            num_layers: Number of stacked reasoning layers
            dropout: Dropout probability
            ffn_ratio: FFN expansion ratio
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # ===== [IMPROVEMENT 1] Enhanced Bridge Projector =====
        # ADDRESSES REVIEWER CONCERN: "桥接的线性瓶颈"
        # Uses Gated Cross-Attention instead of simple MLP for multi-hop reasoning
        self.bridge_projector = GatedBridgeProjector(
            hidden_size=hidden_size,
            num_heads=4,
            dropout=dropout
        )
        
        # ===== [IMPROVEMENT 2] Dynamic Anchor Updater =====
        # ADDRESSES REVIEWER CONCERN: "长距离指代消解能力退化"
        # Updates anchors after each reasoning layer to absorb coreference info
        self.anchor_updater = DynamicAnchorUpdater(
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # ===== [IMPROVEMENT 3] Anisotropy Correction =====
        # ADDRESSES REVIEWER CONCERN: "实体Embedding各向异性"
        # Learnable scale factor for L2-normalized anchors
        self.anchor_scale = nn.Parameter(torch.ones(1) * math.sqrt(hidden_size))
        
        # ===== Cross-Attention Layers (Evidence Retrieval) =====
        self.cross_attn_layers = nn.ModuleList([
            ReasoningLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                ffn_ratio=ffn_ratio
            )
            for _ in range(num_layers)
        ])
        
        # ===== Inter-Slot Communication Layers (Slot Mixing) =====
        # Applied after each cross-attention to allow Bridge to absorb H/T information
        self.slot_comm_layers = nn.ModuleList([
            InterSlotCommunication(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
    
    def precompute_all_kv(self, document: torch.Tensor) -> List[PrecomputedKV]:
        """
        Pre-compute K/V for all layers.
        
        Call this ONCE per document batch (before processing entity pair chunks).
        
        Args:
            document: Document sequence [B, L, H]
        
        Returns:
            List of PrecomputedKV, one per layer
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
        Perform semantic reasoning over document.
        
        Args:
            document: Document token embeddings [B, L, H]
            h_anchors: Head entity embeddings [B, H] (semantic anchor for head query)
            t_anchors: Tail entity embeddings [B, H] (semantic anchor for tail query)
            attention_mask: Padding mask [B, L] (1=valid, 0=padding)
            precomputed_kv_list: Pre-computed K/V for all layers (optional)
        
        Returns:
            h_context: Head-focused context after reasoning [B, H]
            t_context: Tail-focused context after reasoning [B, H]
            bridge_context: Relationship/path context [B, H]
        """
        batch_size = h_anchors.size(0)
        
        # ===== [IMPROVEMENT 3] Anisotropy Correction =====
        # L2 normalize + learnable scale to address representation degeneration
        # Reference: Ethayarajh, 2019 - "How Contextual are Contextualized Word Representations?"
        h_anchors_norm = F.normalize(h_anchors, p=2, dim=-1) * self.anchor_scale
        t_anchors_norm = F.normalize(t_anchors, p=2, dim=-1) * self.anchor_scale
        
        # Keep original for dynamic updates (evolving anchors)
        h_evolving = h_anchors_norm
        t_evolving = t_anchors_norm
        
        # ===== Step 1: Construct Entity-Conditioned Semantic Queries =====
        # [IMPROVEMENT 1] Use GatedBridgeProjector for multi-hop reasoning
        bridge_query_raw = self.bridge_projector(h_anchors_norm, t_anchors_norm)  # [B, H]
        
        # [ANCHOR DROPOUT] Prevent shortcut learning / entity prior exploitation
        # If bridge has complete entity info, it may ignore document and become
        # a pure MLP classifier (Relation Classification instead of Document RE)
        # By dropping some entity signal, we force it to retrieve from document
        if self.training:
            bridge_query_raw = F.dropout(bridge_query_raw, p=0.3, training=True)
        
        bridge_query = bridge_query_raw.unsqueeze(1)  # [B, 1, H]
        
        # Stack anchors as queries: [Head, Tail, Bridge]
        queries = torch.cat([
            h_evolving.unsqueeze(1),  # [B, 1, H] - Head Query (normalized anchor)
            t_evolving.unsqueeze(1),  # [B, 1, H] - Tail Query (normalized anchor)
            bridge_query              # [B, 1, H] - Bridge Query (gated cross-attn)
        ], dim=1)  # [B, 3, H]
        
        # Prepare key_padding_mask for attention
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # Invert: 1->False, 0->True
        
        # ===== Step 2: Iterative Reasoning Loop with Dynamic Anchor Updates =====
        for layer_idx in range(self.num_layers):
            # Get precomputed KV if available
            precomputed_kv = None
            if precomputed_kv_list is not None:
                precomputed_kv = precomputed_kv_list[layer_idx]
            
            # Cross-Attention: Retrieve evidence from document
            queries = self.cross_attn_layers[layer_idx](
                queries, document, key_padding_mask, precomputed_kv
            )
            
            # ===== [IMPROVEMENT 2] Dynamic Anchor Update =====
            # Update Head/Tail anchors with retrieved context to solve
            # long-distance coreference issues (零指代与代词失效问题)
            h_retrieved = queries[:, 0, :]  # Retrieved context for head
            t_retrieved = queries[:, 1, :]  # Retrieved context for tail
            
            # Evolve anchors by fusing original with retrieved
            h_evolving = self.anchor_updater(h_evolving, h_retrieved)
            t_evolving = self.anchor_updater(t_evolving, t_retrieved)
            
            # Update queries with evolved anchors for next layer
            # This allows the anchor to "follow" coreferences across the document
            queries = torch.cat([
                h_evolving.unsqueeze(1),
                t_evolving.unsqueeze(1),
                queries[:, 2:3, :]  # Keep bridge as-is (it already absorbs H/T via inter-slot)
            ], dim=1)
            
            # Inter-Slot Communication: Allow Bridge to absorb H/T information
            queries = self.slot_comm_layers[layer_idx](queries)
        
        # ===== Step 3: Extract Output Contexts =====
        # queries: [B, 3, H] -> split into [Head, Tail, Bridge]
        h_context = queries[:, 0, :]      # [B, H] - Head-focused context (evolved)
        t_context = queries[:, 1, :]      # [B, H] - Tail-focused context (evolved)
        bridge_context = queries[:, 2, :] # [B, H] - Relationship context
        
        return h_context, t_context, bridge_context


def test_semantic_reasoner():
    """Test the new SemanticReasoner architecture."""
    print("=" * 60)
    print("Testing SemanticReasoner (IER Architecture)")
    print("=" * 60)
    
    # Configuration
    batch_size = 4
    seq_len = 128
    hidden_size = 768
    num_layers = 2
    
    # Create module
    reasoner = SemanticReasoner(
        hidden_size=hidden_size,
        num_heads=8,
        num_layers=num_layers,
        dropout=0.1
    )
    
    print(f"\n[Config] batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
    print(f"[Config] num_layers={num_layers}, num_slots=3 (fixed: Head/Tail/Bridge)")
    
    # Create inputs
    document = torch.randn(batch_size, seq_len, hidden_size)
    h_anchors = torch.randn(batch_size, hidden_size)  # Head entity embeddings
    t_anchors = torch.randn(batch_size, hidden_size)  # Tail entity embeddings
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"\n[Input Shapes]")
    print(f"  document: {document.shape}")
    print(f"  h_anchors: {h_anchors.shape}")
    print(f"  t_anchors: {t_anchors.shape}")
    
    # Test without pre-computation
    print(f"\n[Test 1] Forward without KV pre-computation...")
    h_ctx, t_ctx, bridge_ctx = reasoner(document, h_anchors, t_anchors, attention_mask)
    print(f"  ✓ h_context: {h_ctx.shape}")
    print(f"  ✓ t_context: {t_ctx.shape}")
    print(f"  ✓ bridge_context: {bridge_ctx.shape}")
    
    # Test with pre-computation
    print(f"\n[Test 2] Forward with KV pre-computation...")
    precomputed_kv = reasoner.precompute_all_kv(document)
    print(f"  ✓ Pre-computed KV count: {len(precomputed_kv)}")
    print(f"  ✓ KV shape: keys={precomputed_kv[0].keys.shape}, values={precomputed_kv[0].values.shape}")
    
    h_ctx2, t_ctx2, bridge_ctx2 = reasoner(document, h_anchors, t_anchors, attention_mask, precomputed_kv)
    print(f"  ✓ Outputs with precomputed KV: h={h_ctx2.shape}, t={t_ctx2.shape}, bridge={bridge_ctx2.shape}")
    
    # Test gradient flow
    print(f"\n[Test 3] Gradient flow test...")
    loss = h_ctx.sum() + t_ctx.sum() + bridge_ctx.sum()
    loss.backward()
    print(f"  ✓ Gradients computed successfully")
    # Check projector gradients (bridge_token no longer exists, replaced with bridge_projector)
    first_param = next(reasoner.bridge_projector.parameters())
    print(f"  ✓ bridge_projector grad exists: {first_param.grad is not None}")
    
    # Parameter count
    total_params = sum(p.numel() for p in reasoner.parameters())
    print(f"\n[Model Stats]")
    print(f"  Total parameters: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("SemanticReasoner test passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_semantic_reasoner()
