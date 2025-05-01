
"""
Static code analysis for Python error detection.

This implementation uses Python's built-in AST module for static analysis and is from:
- Python AST module documentation (https://docs.python.org/3/library/ast.html)
- Pylint's code analysis techniques (https://github.com/PyCQA/pylint)
- Python's inspect module patterns (https://docs.python.org/3/library/inspect.html)

The error pattern detection approach is by techniques in static analyzers
like Pyflakes (https://github.com/PyCQA/pyflakes) and static analysis research.
"""
import ast
import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

class ErrorPattern:
    """Base class for static error patterns"""
    
    def __init__(self, name, description, fix_suggestion, condescending_remark):
        self.name = name
        self.description = description
        self.fix_suggestion = fix_suggestion
        self.condescending_remark = condescending_remark
    
    def detect(self, code, ast_tree=None):
        """Detect if the pattern exists in the code"""
        return []

class SyntaxErrorPattern(ErrorPattern):
    """Detects common syntax errors"""
    
    def __init__(self):
        super().__init__(
            name="SyntaxError",
            description="Common syntax error detected",
            fix_suggestion="Check syntax carefully, especially for missing colons, parentheses, or indentation issues",
            condescending_remark="Syntax errors are the programming equivalent of typos - small but frustrating."
        )
    
    def detect(self, code, ast_tree=None):
        results = []
        
        # Check for missing colons 
        lines = code.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if re.match(r'^(if|for|while|def|class|else|elif|except|with|try|finally)\s+[^:]*$', line):
                if not line.endswith(':'):
                    results.append({
                        'line': i + 1,
                        'col': len(line),
                        'message': f"Missing colon after '{line.split()[0]}' statement"
                    })
        
        # Check for mismatched parentheses, brackets, etc.
        stack = []
        for i, char in enumerate(code):
            if char in '([{':
                stack.append((char, i))
            elif char in ')]}':
                if not stack:

                    line_idx = code[:i].count('\n')
                    col_idx = i - code[:i].rfind('\n') - 1
                    results.append({
                        'line': line_idx + 1,
                        'col': col_idx,
                        'message': f"Unmatched closing '{char}'"
                    })
                else:
                    last_open, _ = stack.pop()
                    if not ((last_open == '(' and char == ')') or
                            (last_open == '[' and char == ']') or
                            (last_open == '{' and char == '}')):

                        line_idx = code[:i].count('\n')
                        col_idx = i - code[:i].rfind('\n') - 1
                        results.append({
                            'line': line_idx + 1,
                            'col': col_idx,
                            'message': f"Mismatched brackets: '{last_open}' closed with '{char}'"
                        })
        
        # Check for any remaining open brackets
        for open_char, pos in stack:
            line_idx = code[:pos].count('\n')
            col_idx = pos - code[:pos].rfind('\n') - 1 if pos > 0 else 0
            results.append({
                'line': line_idx + 1,
                'col': col_idx,
                'message': f"Unclosed '{open_char}'"
            })
        
        return results

class DivisionByZeroPattern(ErrorPattern):
    """Detects potential division by zero"""
    
    def __init__(self):
        super().__init__(
            name="ZeroDivisionError",
            description="Potential division by zero",
            fix_suggestion="Add a check to ensure the denominator is not zero before division",
            condescending_remark="Even ancient mathematicians knew not to divide by zero. Just saying."
        )
    
    def detect(self, code, ast_tree=None):
        results = []

        try:
            if ast_tree is None:
                ast_tree = ast.parse(code)
        except SyntaxError:
            return results  

        # Track variables that are checked for zero
        safe_vars = set()

        class SafeDivVisitor(ast.NodeVisitor):
            def visit_If(self, node):
                if isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name):
                        var_name = node.test.left.id
                        for op, comparator in zip(node.test.ops, node.test.comparators):
                            if (isinstance(op, (ast.NotEq, ast.Gt, ast.GtE)) and
                                isinstance(comparator, ast.Constant) and comparator.value == 0):
                                safe_vars.add(var_name)
                self.generic_visit(node)

        class DivVisitor(ast.NodeVisitor):
            def visit_BinOp(self, node):
                if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                    # division by constant 0
                    if isinstance(node.right, ast.Constant) and node.right.value == 0:
                        results.append({
                            'line': node.lineno,
                            'col': node.col_offset,
                            'message': "Division by zero detected"
                        })

                    # ivision by variable
                    if isinstance(node.right, ast.Name):
                        var_name = node.right.id
                        if var_name not in safe_vars:
                            results.append({
                                'line': node.lineno,
                                'col': node.col_offset,
                                'message': f"Potential division by zero if '{var_name}' is zero"
                            })

                    # division by len() 
                    if isinstance(node.right, ast.Call) and isinstance(node.right.func, ast.Name) and node.right.func.id == 'len':
                        results.append({
                            'line': node.lineno,
                            'col': node.col_offset,
                            'message': "Potential division by zero if length is zero"
                        })

                self.generic_visit(node)

        # Collect safe variables
        safe_visitor = SafeDivVisitor()
        safe_visitor.visit(ast_tree)

        # Visit divisions
        div_visitor = DivVisitor()
        div_visitor.visit(ast_tree)

        return results


class IndexErrorPattern(ErrorPattern):
    """Detects potential index errors"""
    
    def __init__(self):
        super().__init__(
            name="IndexError",
            description="Potential index out of range",
            fix_suggestion="Check that your indices are valid or use a try/except block",
            condescending_remark="Arrays start at 0 in Python, not 1. I'm looking at you, MATLAB users."
        )
    
    def detect(self, code, ast_tree=None):
        results = []
        
        try:
            if ast_tree is None:
                ast_tree = ast.parse(code)
        except SyntaxError:
            return results 
        
        class IndexVisitor(ast.NodeVisitor):
            def visit_Subscript(self, node):
                
                # Case 1: Using len(x) as an index for x
                if isinstance(node.slice, ast.Call) and isinstance(node.slice.func, ast.Name) and node.slice.func.id == 'len':
                    if len(node.slice.args) == 1 and isinstance(node.slice.args[0], ast.Name) and isinstance(node.value, ast.Name):
                        if node.slice.args[0].id == node.value.id:
                            results.append({
                                'line': node.lineno,
                                'col': node.col_offset,
                                'message': f"Potential index error: {node.value.id}[len({node.value.id})] is out of range"
                            })
                
                # Case 2: Using negative indices without checks
                if isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.op, ast.USub) and isinstance(node.slice.operand, ast.Constant):
                    results.append({
                        'line': node.lineno,
                        'col': node.col_offset,
                        'message': f"Negative index {-node.slice.operand.value} might cause errors if the collection is empty"
                    })
                
                # Case 3: Direct index that might be out of range
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                    if node.slice.value < 0:
                        results.append({
                            'line': node.lineno,
                            'col': node.col_offset,
                            'message': f"Negative index {node.slice.value} might cause errors if the collection is too small"
                        })
                
                self.generic_visit(node)
        
        visitor = IndexVisitor()
        visitor.visit(ast_tree)
        
        return results

# ---------------------- MINIMAX REGRET ----------------------

class MinimaxRegret:
    """Main class for the MinimaxRegret tool"""
    
    def __init__(self, model=None, dataset=None):
        self.model = model
        self.dataset = dataset
        
        # Initialize 
        self.patterns = [
            SyntaxErrorPattern(),
            DivisionByZeroPattern(),
            IndexErrorPattern()
        ]
    
    def analyze_static(self, code):
        """
        Analyze code using static pattern matching
        Returns a list of potential errors
        """
        results = []
        
        # Try to parse the AST
        try:
            ast_tree = ast.parse(code)
            has_syntax_error = False
        except SyntaxError as e:
            results.append({
                'error_type': 'SyntaxError',
                'line': e.lineno,
                'col': e.offset,
                'description': str(e),
                'suggestion': "Fix the syntax error to continue analysis",
                'remark': "Python's parser gave up on your code before I could even start."
            })
            has_syntax_error = True
            ast_tree = None
        
        # Run all patterns
        for pattern in self.patterns:
            if has_syntax_error and pattern.name == "SyntaxError":
                continue
                
            detected_errors = pattern.detect(code, ast_tree)
            
            for error in detected_errors:
                results.append({
                    'error_type': pattern.name,
                    'line': error['line'],
                    'col': error.get('col', 0),
                    'description': f"{pattern.description}: {error['message']}",
                    'suggestion': pattern.fix_suggestion,
                    'remark': pattern.condescending_remark
                })
        
        return results
    
    def analyze_ml(self, code):
        """
        Analyze code using the ML model
        Returns predicted error type, location, and fix suggestion
        """
        if self.model is None or self.dataset is None:
            return None
        
        # Tokenize 
        code_tokens = self.dataset._encode(code)
        code_tensor = torch.tensor([code_tokens], dtype=torch.long).to(next(self.model.parameters()).device)
    
        # predictions
        with torch.no_grad():
            outputs = self.model.forward(code_tensor, None) # type: ignore
    
        # extracting
        error_type_logits = outputs['error_type_logits']
        error_loc_logits = outputs['error_loc_logits']
    
        # predicted error type
        predicted_error_type_idx = torch.argmax(error_type_logits, dim=1).item()
        predicted_error_type = self.dataset.error_types[predicted_error_type_idx]
    
        # predicted error location
        predicted_error_loc = torch.argmax(error_loc_logits, dim=2)[0].tolist()
    
        # token index with highest error probability
        max_error_prob = max(predicted_error_loc)
        error_token_idx = predicted_error_loc.index(max_error_prob)
    
        # map token index to line and column
        code_lines = code.split('\n')
        char_to_line_col = {}
        char_idx = 0
        for line_idx, line in enumerate(code_lines):
            for col_idx, _ in enumerate(line):
                char_to_line_col[char_idx] = (line_idx + 1, col_idx + 1)
                char_idx += 1
            char_to_line_col[char_idx] = (line_idx + 1, len(line) + 1) # End of line
            char_idx += 1
    
        # line and column of error
        error_line, error_col = char_to_line_col.get(error_token_idx, (1, 1))
    
        # fix suggestion
        fixed_tokens = self.model.generate_fix(code_tensor)
        fix_suggestion = self.dataset.decode(fixed_tokens[0].tolist())
    
        # attention weights
        attention_weights = outputs['attention_weights']
    
        return {
            'error_type': predicted_error_type,
            'line': error_line,
            'col': error_col,
            'description': f"Potential {predicted_error_type} detected",
            'suggestion': "Consider this fixed version:\n" + fix_suggestion,
            'attention_weights': attention_weights
        }


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
            # Check if ML model found a unique error type
            ml_error_type = ml_result['error_type']
            if not any(r['error_type'] == ml_error_type for r in static_results):
                results.append(ml_result)
    
    # Sort by line number
    results.sort(key=lambda x: x['line'])
    return results

def visualize_attention(self, code, attention_weights):
    """Generate a visualization of the attention weights"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Process for visualization
    code_lines = code.split('\n')
    code_chars = list(code)
    
    # last layer's attention weights
    last_layer_weights = attention_weights[-1]
    
    # Average
    avg_attention = torch.mean(torch.stack([w[0] for w in last_layer_weights]), dim=0).cpu().numpy()
    
    # heatmap of the attention weights
    plt.figure(figsize=(12, 8))
    plt.imshow(avg_attention, cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.title('Attention Weights for Code Analysis')
    plt.xlabel('Source Token Position')
    plt.ylabel('Target Token Position')
    
    # code tokens on axes
    max_chars = min(50, len(code_chars))
    plt.xticks(np.arange(max_chars), code_chars[:max_chars], rotation=90, fontsize=8)
    plt.yticks(np.arange(max_chars), code_chars[:max_chars], fontsize=8)
    
    plt.tight_layout()
    return plt