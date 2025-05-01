"""
Transformer model implementation for MinimaxRegret.

The following implementation is from:
- Andrej Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT/blob/master/model.py)
- The transformer architecture from "Attention is All You Need" (Vaswani et al., 2017)
- PyTorch's transformer implementations (https://pytorch.org/docs/stable/nn.html#transformer-layers)

Specifically, the attention mechanism is adapted from nanoGPT's CausalSelfAttention class,
modified to use a simplified architecture with fewer parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
    
class AttentionHead(nn.Module):
    """Single head of self-attention"""
    
    def __init__(self, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # applying attention to values
        v = self.value(x)
        out = wei @ v
        return out, wei  

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # collecting outputs and attention weights from all heads
        head_outputs = []
        attn_weights = []
        for head in self.heads:
            output, weights = head(x)
            head_outputs.append(output)
            attn_weights.append(weights)
            
        # concatenate outputs along the dimension
        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, attn_weights

class FeedForward(nn.Module):
    """Simple feed-forward network"""
    
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer block: attention + feed-forward"""
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ff = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, attn_weights = self.attn(x_norm)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights

# MINIMAX REGRET MODEL 

class MinimaxRegretModel(nn.Module):
    """Transformer model for code error prediction"""
    
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=3, block_size=512, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        
        # embedding layers
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout) 
            for _ in range(n_layer)
        ])
        
        # final layer norm and output heads
        self.ln_f = nn.LayerNorm(n_embd)
        
        # multiple output heads for different tasks
        self.error_type_head = nn.Linear(n_embd, 10)  # Predict error category
        self.error_loc_head = nn.Linear(n_embd, block_size)  # Predict error location
        self.fix_suggestion_head = nn.Linear(n_embd, vocab_size)  # Generate fix suggestion
        
        # initialize weights
        self.apply(self._init_weights)
        print(f"Model initialized with {self.get_parameter_count():,} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_parameter_count(self):
        """Count total parameters in the model"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        
        x = self.dropout(tok_emb + pos_emb)
        
        all_attn_weights = []
        
        for block in self.blocks:
            x, attn_weights = block(x)
            all_attn_weights.append(attn_weights)
        
        x = self.ln_f(x)
        
        # predictions from different heads
        error_type_logits = self.error_type_head(x[:, -1, :])  
        error_loc_logits = self.error_loc_head(x)  
        fix_logits = self.fix_suggestion_head(x)  
        
        # calculates loss if targets are provided
        loss = None
        if targets is not None:
            error_type_targets, error_loc_targets, fix_targets = targets
            
            error_type_loss = F.cross_entropy(error_type_logits, error_type_targets)
            error_loc_loss = F.cross_entropy(error_loc_logits.view(-1, T), error_loc_targets.view(-1))
            fix_loss = F.cross_entropy(fix_logits.view(-1, fix_logits.size(-1)), fix_targets.view(-1))

            loss = 0.5 * error_type_loss + 0.2 * error_loc_loss + 0.3 * fix_loss
        
        return {
            'error_type_logits': error_type_logits,
            'error_loc_logits': error_loc_logits,
            'fix_logits': fix_logits,
            'attention_weights': all_attn_weights,
            'loss': loss
        }
    def generate_fix(self, code_tokens, max_new_tokens=100, temperature=0.8):
        """Generate a fix suggestion for the code"""
        if not isinstance(code_tokens, torch.Tensor):
            code_tokens = torch.tensor([code_tokens], dtype=torch.long).to(next(self.parameters()).device)
    
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(code_tokens, None)
                fix_logits = outputs['fix_logits']
        
            logits = fix_logits[:, -1, :] / temperature
        
            probs = F.softmax(logits, dim=-1)
        
            idx_next = torch.multinomial(probs, num_samples=1)
        
            code_tokens = torch.cat((code_tokens, idx_next), dim=1)
        
            if idx_next.item() == 0:  
                break
    
        return code_tokens
    
    def analyze(self, code):
        """
        Analyze code using both static pattern matching and ML model
        Returns a combined list of potential errors
        """
        results = []
        
        # static analysis
        static_results = self.analyze_static(code)
        results.extend(static_results)
        
        # ML analysis
        if self.model is not None and self.dataset is not None:
            ml_result = self.analyze_ml(code)
            if ml_result:
                ml_error_type = ml_result['error_type']
                if not any(r['error_type'] == ml_error_type for r in static_results):
                    results.append(ml_result)
        
        results.sort(key=lambda x: x['line'])
        return results

    def visualize_attention(self, code, attention_weights):
        """Generate a visualization of the attention weights"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        code_lines = code.split('\n')
        code_chars = list(code)
        
        last_layer_weights = attention_weights[-1]
        
        avg_attention = torch.mean(torch.stack([w[0] for w in last_layer_weights]), dim=0).cpu().numpy()
        
        # heatmap of the attention weights
        plt.figure(figsize=(12, 8))
        plt.imshow(avg_attention, cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.title('Attention Weights for Code Analysis')
        plt.xlabel('Source Token Position')
        plt.ylabel('Target Token Position')
        
        max_chars = min(50, len(code_chars))
        plt.xticks(np.arange(max_chars), code_chars[:max_chars], rotation=90, fontsize=8)
        plt.yticks(np.arange(max_chars), code_chars[:max_chars], fontsize=8)
        
        plt.tight_layout()
        return plt