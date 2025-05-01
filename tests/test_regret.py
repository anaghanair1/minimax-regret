#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for MinimaxRegret

This script contains unit tests for the MinimaxRegret tool.
"""

import sys
import os
import unittest
import torch

# parent directory 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MinimaxRegretModel
from src.dataset import CodeErrorDataset, ErrorExample
from src.analysis import MinimaxRegret, SyntaxErrorPattern, DivisionByZeroPattern, IndexErrorPattern


class TestErrorPatterns(unittest.TestCase):
    """Test the static error pattern detection"""
    
    def setUp(self):
        """Set up error patterns for testing"""
        self.syntax_pattern = SyntaxErrorPattern()
        self.division_pattern = DivisionByZeroPattern()
        self.index_pattern = IndexErrorPattern()
    
    def test_syntax_error_detection(self):
        """Test detection of syntax errors"""
        # Missing colon
        code = "if True\n    print('Hello')"
        errors = self.syntax_pattern.detect(code)
        self.assertTrue(len(errors) > 0, "Should detect missing colon")
        self.assertEqual(errors[0]['line'], 1, "Error should be on line 1")
        
        # Correct syntax
        code = "if True:\n    print('Hello')"
        errors = self.syntax_pattern.detect(code)
        self.assertEqual(len(errors), 0, "Should not detect error in correct code")
    
    def test_division_by_zero_detection(self):
        """Test detection of division by zero"""
        # Direct division by zero
        code = "result = 10 / 0"
        errors = self.division_pattern.detect(code)
        self.assertTrue(len(errors) > 0, "Should detect division by zero")
        
        # Division by variable
        code = "def divide(a, b):\n    return a / b"
        errors = self.division_pattern.detect(code)
        self.assertTrue(len(errors) > 0, "Should detect potential division by variable")
        
        # Safe division
        code = "def divide(a, b):\n    if b != 0:\n        return a / b"
        errors = self.division_pattern.detect(code)
        self.assertEqual(len(errors), 0, "Should not detect error in safe division")
    
    def test_index_error_detection(self):
        """Test detection of index errors"""
        # Using len(x) as index
        code = "def get_last(items):\n    return items[len(items)]"
        errors = self.index_pattern.detect(code)
        self.assertTrue(len(errors) > 0, "Should detect index error with len(items)")
        
        # Safe indexing
        code = "def get_last(items):\n    if items:\n        return items[len(items) - 1]"
        errors = self.index_pattern.detect(code)
        self.assertEqual(len(errors), 0, "Should not detect error in safe indexing")


class TestMinimaxRegret(unittest.TestCase):
    """Test the MinimaxRegret class"""
    
    def setUp(self):
        """Set up MinimaxRegret for testing"""
        self.regret = MinimaxRegret()
    
    def test_static_analysis(self):
        """Test static pattern matching analysis"""
        # Test with code containing a syntax error
        code = "if True\n    print('Hello')"
        results = self.regret.analyze_static(code)
        self.assertTrue(len(results) > 0, "Should detect syntax error")
        self.assertEqual(results[0]['error_type'], "SyntaxError", "Should identify error type")
        
        # Test with multiple errors
        code = "def bad_func():\n    x = 10 / 0\n    return items[len(items)]"
        results = self.regret.analyze_static(code)
        self.assertTrue(len(results) >= 2, "Should detect multiple errors")
        
        # Test with correct code
        code = "def good_func():\n    x = 10\n    return x"
        results = self.regret.analyze_static(code)
        self.assertEqual(len(results), 0, "Should not detect errors in correct code")


class TestDataset(unittest.TestCase):
    """Test the dataset functionality"""
    
    def test_sample_dataset(self):
        """Test the sample dataset creation"""
        dataset = CodeErrorDataset()
        self.assertTrue(len(dataset) > 0, "Dataset should contain examples")
        
        # Test getting an item
        x, targets = dataset[0]
        self.assertIsInstance(x, torch.Tensor, "Input should be a tensor")
        self.assertTrue(len(targets) >= 2, "Targets should contain multiple elements")
    
    def test_encoding_decoding(self):
        """Test encoding and decoding functionality"""
        dataset = CodeErrorDataset()
        
        # encode and decode
        test_code = "def test():\n    return 42"
        encoded = dataset._encode(test_code)
        decoded = dataset.decode(encoded)
        self.assertEqual(test_code, decoded, "Encoding and decoding should preserve code")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestModel(unittest.TestCase):
    """Test the model functionality (GPU only)"""
    
    def setUp(self):
        """Set up model for testing"""
        dataset = CodeErrorDataset()
        self.model = MinimaxRegretModel(
            vocab_size=dataset.vocab_size,
            n_embd=64,
            n_head=2,
            n_layer=2,
            block_size=256,
            dropout=0.1
        ).to('cuda')
    
    def test_forward_pass(self):
        """Test forward pass through the model"""
        x = torch.ones((1, 10), dtype=torch.long).to('cuda')
        
        # Run forward pass
        outputs = self.model(x)
        
        self.assertIn('error_type_logits', outputs, "Should output error type logits")
        self.assertIn('error_loc_logits', outputs, "Should output error location logits")
        self.assertIn('fix_logits', outputs, "Should output fix logits")
    
    def test_generate_fix(self):
        """Test fix generation"""
        x = torch.ones((1, 10), dtype=torch.long).to('cuda')
        
        # Generate fix
        fixed = self.model.generate_fix(x, max_new_tokens=5)
        
        self.assertIsInstance(fixed, torch.Tensor, "Fix should be a tensor")
        self.assertTrue(fixed.size(1) > 10, "Fix should add tokens")


if __name__ == "__main__":
    unittest.main()