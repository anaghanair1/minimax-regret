"""
Visualization utilities for transformer attention.

The visualization techniques are from:
- Hugging Face Transformers attention visualization (https://huggingface.co/docs/transformers/attentions_visualization)
- "Attention is All You Need" paper visualizations (Vaswani et al., 2017)
- Matplotlib heatmap examples (https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html)
- BertViz (https://github.com/jessevig/bertviz) for transformer visualization approaches
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention(self, code, attention_weights):
        """Generate a visualization of the attention weights"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Process the code 
        code_lines = code.split('\n')
        code_chars = list(code)
        
        # Takes the last layer's attention weights
        last_layer_weights = attention_weights[-1]
        
        # Average across heads
        avg_attention = torch.mean(torch.stack([w[0] for w in last_layer_weights]), dim=0).cpu().numpy()
        
        # creates a heatmap of the attention weights
        plt.figure(figsize=(12, 8))
        plt.imshow(avg_attention, cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.title('Attention Weights for Code Analysis')
        plt.xlabel('Source Token Position')
        plt.ylabel('Target Token Position')
        
        # Adds code tokens on axes 
        max_chars = min(50, len(code_chars))
        plt.xticks(np.arange(max_chars), code_chars[:max_chars], rotation=90, fontsize=8)
        plt.yticks(np.arange(max_chars), code_chars[:max_chars], fontsize=8)
        
        plt.tight_layout()
        return plt