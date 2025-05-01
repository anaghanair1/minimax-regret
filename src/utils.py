# Utility functions

import os
import argparse
import torch
from typing import Dict, List

from .dataset import CodeErrorDataset
from .model import MinimaxRegretModel
from .analysis import MinimaxRegret

def format_result(result):
    """Format a single result for display"""
    return (
        f"Line {result['line']}, Col {result.get('col', 0)} - {result['error_type']}: {result['description']}\n"
        f"    Suggestion: {result['suggestion']}\n"
        f"    Note: {result.get('remark', '')}\n"
    )

def main():
    """Main function to demonstrate the model"""
    parser = argparse.ArgumentParser(description='MinimaxRegret - Python Error Predictor')
    parser.add_argument('file', nargs='?', type=str, help='Python file to analyze')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--model', '-m', action='store_true', help='Use ML model for prediction')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize 
    if args.model:
        dataset = CodeErrorDataset()
        
        model = MinimaxRegretModel(
            vocab_size=dataset.vocab_size,
            n_embd=128,       # Embedding dimension
            n_head=4,         # Number of attention heads
            n_layer=3,        # Number of transformer layers
            block_size=512,   # Maximum sequence length
            dropout=0.1
        ).to(device)
        
        model_path = 'minimax_regret_model.pt'
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("Model file not found. Using static analysis only.")
            model = None
        
        regret = MinimaxRegret(model, dataset)
    else:
        regret = MinimaxRegret()
    
    if args.file:
        try:
            with open(args.file, 'r') as f:
                code = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("No file specified. Enter your code below (Ctrl+D or Ctrl+Z to finish):")
        code_lines = []
        try:
            while True:
                code_lines.append(input())
        except EOFError:
            code = '\n'.join(code_lines)
    
    # analyze code
    results = regret.analyze(code)
    
    # display results
    if results:
        print("\nPotential issues detected:\n")
        for result in results:
            print(format_result(result))
        print(f"Found {len(results)} potential issues.")
    else:
        print("No issues detected! (But that doesn't mean your code works...)")


if __name__ == "__main__":
    main()