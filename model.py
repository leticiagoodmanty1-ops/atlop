"""
DocRE Model with Entity-Guided Evidence Accumulation (IER Architecture).

Core Innovation:
    - SemanticReasoner: Fixed 3 semantic slots (Head/Tail/Bridge) instead of random slots
    - Dual-Path Scoring: Score_Direct (Bilinear H-T) + Score_Reasoning (Bridge)
    - No auxiliary losses: No orthogonality loss, no gate balancing

Architecture:
    1. BERT/RoBERTa Encoder: Produces document representations
    2. Entity Extraction: LogSumExp pooling for entity mentions
    3. SemanticReasoner: Entity-guided evidence accumulation
    4. Dual Scoring:
       - Score_Direct: Bilinear(h_context, t_context) for direct entity pair features
       - Score_Reasoning: Linear(bridge_context) for reasoning path evidence
    5. Final Logits: Score_Direct + Score_Reasoning

Author: Refactored to IER architecture
Performance: Optimized with KV pre-computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
from reasoning_layer import SemanticReasoner, PrecomputedKV
from typing import Optional, Tuple, List, Dict, Any


class DocREModel(nn.Module):
    """
    Document-level Relation Extraction Model with Entity-Guided Evidence Accumulation.
    
    PERFORMANCE OPTIMIZATIONS:
        1. KV Pre-computation: Document K/V computed once, reused for all chunks
        2. Efficient Chunking: Process entity pairs in batches
        3. Gradient Checkpointing: Enabled on BERT encoder
    
    ARCHITECTURE INNOVATION:
        Instead of random slot initialization + orthogonality loss (prone to mode collapse),
        we use entity embeddings as semantic anchors:
        
        - Head_Query: Focuses on evidence related to head entity
        - Tail_Query: Focuses on evidence related to tail entity  
        - Bridge_Query: Learns to find relationship/path evidence
    """
    
    def __init__(
        self,
        config,
        model,
        emb_size: int = 768,
        block_size: int = 64,
        num_labels: int = -1,
        # Reasoning parameters
        num_reasoning_layers: int = 2,
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        # ===== Dual-Path Classification Head =====
        # Path 1: Bilinear for entity pair interaction (direct evidence)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        
        # Path 2: Bridge classifier for reasoning path evidence
        # CRITICAL: Bridge needs direct supervision signal, not just second-order gradients
        # This allows Bridge to specialize in finding long-range contextual evidence
        self.bridge_classifier = nn.Linear(emb_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        
        # ===== Semantic Reasoner (IER Architecture) =====
        # Uses entity embeddings as anchors + entity-conditioned bridge
        self.reasoner = SemanticReasoner(
            hidden_size=config.hidden_size,
            num_heads=8,
            num_layers=num_reasoning_layers,
            dropout=0.1,
            ffn_ratio=4
        )
        
        # ===== Context Projector =====
        # Project reasoner output to embedding size for bilinear classifier
        # This replaces the old head_extractor/tail_extractor with a cleaner design
        self.context_proj = nn.Linear(config.hidden_size, emb_size)
        
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # ===== Logging/Debugging Attributes =====
        self.last_loss = 0.0

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence using pretrained transformer."""
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(
            self.model, input_ids, attention_mask, start_tokens, end_tokens
        )
        return sequence_output, attention

    def get_hrt(
        self,
        sequence_output: torch.Tensor,
        attention: torch.Tensor,
        entity_pos: List,
        hts: List,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract head/tail entity representations and perform semantic reasoning.
        
        CORE CHANGE FROM OLD ARCHITECTURE:
            - OLD: Random slot initialization -> Orthogonality Loss -> AttentionPooling
            - NEW: Entity embeddings as semantic anchors -> Bridge learns relationships
        
        PERFORMANCE OPTIMIZATION:
            Pre-computes K/V for all reasoning layers ONCE before chunking,
            then reuses them for each entity pair chunk.
        
        Returns:
            h_context: Head-focused context after reasoning [N_rel, H]
            t_context: Tail-focused context after reasoning [N_rel, H]
            bridge_context: Relationship/path context [N_rel, H]
        """
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        
        hss, tss = [], []
        num_relations_per_doc = []
        
        # ===== Step 1: Extract Entity Representations =====
        # These will be used as semantic anchors for the reasoner
        for i in range(len(entity_pos)):
            entity_embs = []
            
            for e in entity_pos[i]:
                if len(e) > 1:
                    # Multiple mentions: aggregate with LogSumExp
                    e_emb_list = []
                    for start, end in e:
                        if start + offset < c:
                            e_emb_list.append(sequence_output[i, start + offset])
                    if len(e_emb_list) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb_list, dim=0), dim=0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                else:
                    # Single mention
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                entity_embs.append(e_emb)

            entity_embs = torch.stack(entity_embs, dim=0)

            if len(hts[i]) == 0:
                num_relations_per_doc.append(0)
                continue
            
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            if ht_i.dim() == 1:
                ht_i = ht_i.unsqueeze(0)
            
            # Get head and tail entity embeddings
            # These are the semantic anchors for the reasoner
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])  # [N_rel_i, H]
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])  # [N_rel_i, H]
            
            hss.append(hs)
            tss.append(ts)
            num_relations_per_doc.append(len(hts[i]))
        
        # Stack all entity pairs
        hss = torch.cat(hss, dim=0)  # [N_rel_total, H]
        tss = torch.cat(tss, dim=0)  # [N_rel_total, H]
        total_relations = hss.size(0)
        
        # ===== Step 2: Semantic Reasoning with KV Pre-computation =====
        doc_indices = torch.tensor(
            [doc_idx for doc_idx, count in enumerate(num_relations_per_doc) for _ in range(count)],
            device=sequence_output.device
        )
        
        # Chunking thresholds
        PARALLEL_THRESHOLD = 128
        CHUNK_SIZE = 64
        
        if total_relations <= PARALLEL_THRESHOLD:
            # === Fast Path: Single parallel forward pass ===
            expanded_seq = sequence_output[doc_indices]
            expanded_mask = attention_mask[doc_indices] if attention_mask is not None else None
            
            # Pre-compute K/V for all layers (optimization)
            precomputed_kv = self.reasoner.precompute_all_kv(expanded_seq)
            
            # Pass entity embeddings as semantic anchors
            h_context, t_context, bridge_context = self.reasoner(
                expanded_seq, hss, tss, expanded_mask, precomputed_kv
            )
            
            del expanded_seq, expanded_mask, precomputed_kv
        else:
            # === Memory-Safe Path: Chunked processing with KV reuse ===
            all_h_contexts = []
            all_t_contexts = []
            all_bridge_contexts = []
            
            # Process by document to maximize KV reuse
            relation_idx = 0
            for doc_idx, count in enumerate(num_relations_per_doc):
                if count == 0:
                    continue
                
                # Get document-specific data
                doc_seq = sequence_output[doc_idx:doc_idx+1]  # [1, L, H]
                doc_mask = attention_mask[doc_idx:doc_idx+1] if attention_mask is not None else None
                
                # Pre-compute K/V once for this document
                precomputed_kv = self.reasoner.precompute_all_kv(doc_seq)
                
                # Get anchors for this document
                doc_hs = hss[relation_idx:relation_idx + count]  # [count, H]
                doc_ts = tss[relation_idx:relation_idx + count]  # [count, H]
                
                for chunk_start in range(0, count, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, count)
                    chunk_hs = doc_hs[chunk_start:chunk_end]
                    chunk_ts = doc_ts[chunk_start:chunk_end]
                    chunk_size = chunk_hs.size(0)
                    
                    # Expand document to match chunk size
                    expanded_seq = doc_seq.expand(chunk_size, -1, -1)
                    expanded_mask = doc_mask.expand(chunk_size, -1) if doc_mask is not None else None
                    
                    # Expand precomputed KV
                    expanded_kv = [
                        PrecomputedKV(
                            keys=kv.keys.expand(chunk_size, -1, -1),
                            values=kv.values.expand(chunk_size, -1, -1)
                        )
                        for kv in precomputed_kv
                    ]
                    
                    # Semantic reasoning
                    h_ctx, t_ctx, bridge_ctx = self.reasoner(
                        expanded_seq, chunk_hs, chunk_ts, expanded_mask, expanded_kv
                    )
                    
                    all_h_contexts.append(h_ctx)
                    all_t_contexts.append(t_ctx)
                    all_bridge_contexts.append(bridge_ctx)
                    
                    del expanded_seq, expanded_mask, expanded_kv
                
                del precomputed_kv
                relation_idx += count
            
            h_context = torch.cat(all_h_contexts, dim=0)
            t_context = torch.cat(all_t_contexts, dim=0)
            bridge_context = torch.cat(all_bridge_contexts, dim=0)
            
            del all_h_contexts, all_t_contexts, all_bridge_contexts
        
        return h_context, t_context, bridge_context

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[List] = None,
        entity_pos: Optional[List] = None,
        hts: Optional[List] = None,
        instance_mask: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        Forward pass for training and inference.
        
        Dual-Path Scoring Architecture:
            1. Score_Direct = Bilinear(h_context, t_context)
               - Captures explicit entity pair features
            2. Score_Reasoning = Linear(bridge_context)
               - Captures implicit reasoning path evidence
               - Bridge has direct supervision to learn what evidence helps classification
            3. Final = Score_Direct + Score_Reasoning
               - Simple relations use direct path; complex relations benefit from bridge
        """
        # Encode input sequence
        sequence_output, attention = self.encode(input_ids, attention_mask)
        
        # Semantic reasoning: get all three contexts
        # CRITICAL: Don't discard bridge_context - it needs direct supervision!
        h_context, t_context, bridge_context = self.get_hrt(
            sequence_output, attention, entity_pos, hts, attention_mask
        )

        # ===== Dual-Path Classification =====
        
        # Path 1: Direct entity pair features (Bilinear)
        h_proj = torch.tanh(self.context_proj(h_context))  # [N_rel, emb_size]
        t_proj = torch.tanh(self.context_proj(t_context))  # [N_rel, emb_size]
        
        # Block-wise bilinear (parameter efficient)
        b1 = h_proj.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = t_proj.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits_direct = self.bilinear(bl)  # [N_rel, num_labels]
        
        # Path 2: Reasoning path evidence (Bridge classifier)
        # Bridge gets direct gradient signal, forcing it to find useful evidence
        bridge_proj = torch.tanh(self.context_proj(bridge_context))  # [N_rel, emb_size]
        logits_reasoning = self.bridge_classifier(bridge_proj)  # [N_rel, num_labels]
        
        # Fusion: both paths contribute to final prediction
        # - Simple relations: dominated by logits_direct
        # - Complex multi-hop relations: logits_reasoning provides residual correction
        logits = logits_direct + logits_reasoning

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        
        if labels is not None:
            # Process labels
            processed_labels = []
            for label in labels:
                if isinstance(label, torch.Tensor):
                    processed_labels.append(label)
                else:
                    processed_labels.append(torch.tensor(label))
            labels = torch.cat(processed_labels, dim=0).to(logits)
            
            # Classification Loss (ATLoss)
            # No auxiliary losses needed - the semantic anchor design prevents mode collapse
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
            
            # Store for logging
            self.last_loss = loss.item()
        
        return output
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for monitoring training."""
        return {
            'loss': self.last_loss,
        }
