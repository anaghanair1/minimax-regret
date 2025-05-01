"""
Dataset implementation for code error examples.

The dataset implementation is by:
- PyTorch's Dataset class (https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
- HuggingFace's tokenization approaches (https://huggingface.co/docs/transformers/tokenizer_summary)
- CodeBERT's dataset preparation (https://github.com/microsoft/CodeBERT)

The error examples are common Python errors documented in:
- Python Documentation (https://docs.python.org/3/library/exceptions.html)
- Common errors found on Stack Overflow
"""
# Sample dataset creation function

import torch
import json
import os
import ast
import re
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

class ErrorExample:
    """Represents a code example with error information"""
    
    def __init__(self, code, error_type, error_message, error_locations, fixed_code):
        self.code = code
        self.error_type = error_type
        self.error_message = error_message
        self.error_locations = error_locations 
        self.fixed_code = fixed_code

class CodeErrorDataset(Dataset):
    """Dataset for code error examples with fixes"""
    
    def __init__(self, data_path='data/error_examples.json', max_len=512):
        self.max_len = max_len
        
        # Create directory 
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Use a pre-defined dataset 
        if not os.path.exists(data_path):
            print("Creating sample dataset...")
            examples = self._create_sample_dataset()
            
            # sample dataset for future
            with open(data_path, 'w') as f:
                json.dump(examples, f, indent=2)
            
            self.examples = [ErrorExample(**ex) for ex in examples]
        else:
            with open(data_path, 'r') as f:
                examples = json.load(f)
            self.examples = [ErrorExample(**ex) for ex in examples]
        
        # character-level tokenizer
        all_text = []

        for ex in self.examples:
            all_text.append(ex.code)
            all_text.append(ex.fixed_code)
            all_text.append(ex.error_message)

        extra_test_chars = "0123456789\n:()[]{}+-*/= '\"_."  
        all_text.append(extra_test_chars)

        self.chars = sorted(list(set(''.join(all_text))))
        self.vocab_size = len(self.chars) + 1  
        
        # character mappings
        self.stoi = {ch: i+1 for i, ch in enumerate(self.chars)}  
        self.itos = {i+1: ch for i, ch in enumerate(self.chars)}
        self.itos[0] = '<END>'
        
        # error type mapping
        self.error_types = sorted(list(set(ex.error_type for ex in self.examples)))
        self.error_type_to_idx = {error_type: i for i, error_type in enumerate(self.error_types)}
        
        print(f"Dataset loaded with {len(self.examples)} examples and vocabulary size {self.vocab_size}")
        print(f"Error types: {', '.join(self.error_types)}")
    
    def _create_sample_dataset(self):
        """Create a small dataset of common Python errors"""
        return [
            {
                "code": "def greet(name):\n    if name == 'Alice'\n        return 'Hello, Alice!'\n    else:\n        return 'Hello, stranger!'",
                "error_type": "SyntaxError",
                "error_message": "SyntaxError: expected ':'",
                "error_locations": [(2, 23)],
                "fixed_code": "def greet(name):\n    if name == 'Alice':\n        return 'Hello, Alice!'\n    else:\n        return 'Hello, stranger!'"
            },
            {
                "code": "def calculate_average(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total / len(numbers)",
                "error_type": "ZeroDivisionError",
                "error_message": "ZeroDivisionError: division by zero",
                "error_locations": [(5, 11)],
                "fixed_code": "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    total = 0\n    for num in numbers:\n        total += num\n    return total / len(numbers)"
            },
            {
                "code": "def get_last_item(items):\n    return items[len(items)]",
                "error_type": "IndexError",
                "error_message": "IndexError: list index out of range",
                "error_locations": [(2, 11)],
                "fixed_code": "def get_last_item(items):\n    if not items:\n        return None\n    return items[len(items) - 1]"
            },
            {
                "code": "x = 10\ny = '20'\nresult = x + y",
                "error_type": "TypeError",
                "error_message": "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
                "error_locations": [(3, 8)],
                "fixed_code": "x = 10\ny = '20'\nresult = x + int(y)  # Convert string to int"
            },
            {
                "code": "def recursive_function(n):\n    if n <= 0:\n        return 1\n    return n * recursive_function(n-1)",
                "error_type": "RecursionError",
                "error_message": "RecursionError: maximum recursion depth exceeded",
                "error_locations": [(4, 15)],
                "fixed_code": "def recursive_function(n):\n    if n <= 0:\n        return 1\n    # Add a limit to prevent stack overflow\n    if n > 1000:\n        raise ValueError('Input too large, would cause recursion error')\n    return n * recursive_function(n-1)"
            },
            {
                "code": "data = {'name': 'Alice', 'age': 30}\nprint(data['email'])",
                "error_type": "KeyError",
                "error_message": "KeyError: 'email'",
                "error_locations": [(2, 11)],
                "fixed_code": "data = {'name': 'Alice', 'age': 30}\nif 'email' in data:\n    print(data['email'])\nelse:\n    print('Email not found')"
            },
            {
                "code": "import non_existent_module",
                "error_type": "ModuleNotFoundError",
                "error_message": "ModuleNotFoundError: No module named 'non_existent_module'",
                "error_locations": [(1, 7)],
                "fixed_code": "try:\n    import non_existent_module\nexcept ModuleNotFoundError:\n    print('Module not found, installing or using alternative...')\n    # Use alternative module or install the module"
            },
            {
                "code": "x = [1, 2, 3]\ndel x\nprint(x)",
                "error_type": "NameError",
                "error_message": "NameError: name 'x' is not defined",
                "error_locations": [(3, 6)],
                "fixed_code": "x = [1, 2, 3]\nprint(x)  # Print before deletion\ndel x  # Move deletion after use"
            },
            {
                "code": "def process_data():\n    data = get_data()\n    return analyze(data)",
                "error_type": "NameError",
                "error_message": "NameError: name 'get_data' is not defined",
                "error_locations": [(2, 11)],
                "fixed_code": "def get_data():\n    # Implementation of get_data\n    return [1, 2, 3]\n\ndef analyze(data):\n    # Implementation of analyze\n    return sum(data)\n\ndef process_data():\n    data = get_data()\n    return analyze(data)"
            },
            {
                "code": "def divide(a, b):\n    return a / b\n\nresult = divide(10, 0)",
                "error_type": "ZeroDivisionError",
                "error_message": "ZeroDivisionError: division by zero",
                "error_locations": [(2, 13)],
                "fixed_code": "def divide(a, b):\n    if b == 0:\n        return 'Cannot divide by zero'\n    return a / b\n\nresult = divide(10, 0)"
            }
        ]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        code_encoded = self._encode(example.code)
        fixed_encoded = self._encode(example.fixed_code)
        
        # error location mask
        error_loc = torch.zeros(self.max_len, dtype=torch.long)
        
        # heuristic to identify error location in token space
        code_lines = example.code.split('\n')
        for line, col in example.error_locations:
            if line < len(code_lines):
                tokens_before_error = 0
                for i in range(line):
                    tokens_before_error += len(code_lines[i]) + 1  # +1 for newline
                tokens_before_error += col
                
                error_loc_idx = min(tokens_before_error, self.max_len - 1)
                error_loc[error_loc_idx] = 1
        
        # error type index
        error_type_idx = self.error_type_to_idx[example.error_type]
        
        # convert to tensors
        x = torch.zeros(self.max_len, dtype=torch.long)
        encoded = self._encode(example.code)
        x[:min(len(encoded), self.max_len)] = torch.tensor(encoded[:self.max_len])

        error_type_target = torch.tensor(error_type_idx, dtype=torch.long)
        error_loc_target = error_loc
        fix_target = torch.tensor(fixed_encoded, dtype=torch.long)
        
        return x, (error_type_target, error_loc_target)

    
    def _encode(self, text):
        """Convert text to token ids"""
        return [self.stoi.get(c, 1) for c in text[:self.max_len]]  

    def decode(self, ids):
        """Convert token ids back to text"""
        return ''.join([self.itos.get(i, '') for i in ids if i > 0])