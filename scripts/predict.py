"""
Prediction script.

This command-line interface is from:
- Click's CLI design patterns (https://click.palletsprojects.com/)
- Pylint's command-line interface (https://github.com/PyCQA/pylint)
- Python's argparse standard library (https://docs.python.org/3/library/argparse.html)
- Error reporting approaches from compiler design literature
"""

import sys
import os
import argparse
import torch

# parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import MinimaxRegret
from src.dataset import CodeErrorDataset
from src.model import MinimaxRegretModel
from src.utils import format_result

def main():
    """Main entry point for the prediction script"""
    parser = argparse.ArgumentParser(description='MinimaxRegret - Python Error Predictor')
    parser.add_argument('file', nargs='?', type=str, help='Python file to analyze')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--model', '-m', action='store_true', help='Use ML model for prediction')
    args = parser.parse_args()
    
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize
    if args.model:
        # dataset
        dataset = CodeErrorDataset()
        
        # model
        model = MinimaxRegretModel(
            vocab_size=dataset.vocab_size,
            n_embd=128,       # Embedding dimension
            n_head=4,         # Number of attention heads
            n_layer=3,        # Number of transformer layers
            block_size=512,   # Maximum sequence length
            dropout=0.1
        ).to(device)
        
        # Load model
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'minimax_regret_model.pt')
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("Model file not found. Using static analysis only.")
            model = None
        
        regret = MinimaxRegret(model, dataset)
    else:
        regret = MinimaxRegret()

    # Get code 
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
                line = input()
                code_lines.append(line)
        except EOFError:
            code = '\n'.join(code_lines)

    # static analysis
    static_results = regret.analyze_static(code)
    results = static_results
    
    # ML analysis
    if args.model and regret.model is not None and regret.dataset is not None:
        ml_result = regret.analyze_ml(code)
        if ml_result:
            # Check if ML model found a unique error type
            ml_error_type = ml_result['error_type']
            if not any(r['error_type'] == ml_error_type for r in static_results):
                results.append(ml_result)
    
    # sorting by line number
    results.sort(key=lambda x: x['line'])

    if results:
        print("\nPotential issues detected:\n")
        for result in results:
            print(format_result(result))
        
        if args.verbose:
            # print additional information
            ml_results = [r for r in results if 'attention_weights' in r]
            if ml_results and args.model:
                print("\nML Model Analysis:")
                print(f"Error type: {ml_results[0].get('error_type', 'Unknown')}")
                print(f"Line: {ml_results[0].get('line', 'Unknown')}")
                print(f"Column: {ml_results[0].get('col', 'Unknown')}")
                
                # print fix
                if 'suggestion' in ml_results[0]:
                    fix_suggestion = ml_results[0]['suggestion']
                    if fix_suggestion.startswith("Consider this fixed version:\n"):
                        fixed_code = fix_suggestion.replace("Consider this fixed version:\n", "")
                        print("\nFixed code:")
                        print("===========")
                        print(fixed_code)
        
        print(f"\nFound {len(results)} potential issues.")
    else:
        print("No issues detected! (But that doesn't mean your code works...)")

if __name__ == "__main__":
    main()