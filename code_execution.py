"""
Utility functions for safely executing Python code and running tests.
"""

import ast
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional


def check_compilation(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check if Python code compiles without syntax errors.
    
    Args:
        code: Python code string to check
        
    Returns:
        Tuple of (compiles: bool, error_message: Optional[str])
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        return False, f"ParseError: {str(e)}"


def extract_code_from_completion(completion: str, starter_code: str = "") -> str:
    """
    Extract Python code from model completion.
    
    This function attempts to extract the actual code from the model's output,
    which might include markdown code blocks, explanations, etc.
    
    Args:
        completion: Model's raw completion text
        starter_code: Optional starter code to prepend
        
    Returns:
        Extracted Python code string
    """
    # Strip Qwen-style chat markers and special tokens if they leak into the output
    for marker in (
        "<|im_start|>system",
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
    ):
        completion = completion.replace(marker, "")

    # Truncate at Qwen FIM/file tokens - everything after these is garbage
    for stop_token in (
        "<|file_sep|>",
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|repo_name|>",
    ):
        if stop_token in completion:
            completion = completion[:completion.find(stop_token)]

    # Remove markdown code blocks if present
    if "```python" in completion:
        # Extract code between ```python and ```
        start_idx = completion.find("```python") + len("```python")
        end_idx = completion.find("```", start_idx)
        if end_idx != -1:
            completion = completion[start_idx:end_idx].strip()
    elif "```" in completion:
        # Generic code block
        start_idx = completion.find("```") + 3
        end_idx = completion.find("```", start_idx)
        if end_idx != -1:
            completion = completion[start_idx:end_idx].strip()
    
    # Post-process: Add typing imports if type annotations are used
    extracted_code = completion.strip()
    
    # Check if code uses type annotations that need imports
    needs_typing = any(annotation in extracted_code for annotation in
                       ['List[', 'Dict[', 'Tuple[', 'Optional[', 'Any', 'Union[', 'Set['])

    if needs_typing and 'from typing import' not in extracted_code:
        # Determine which typing imports are needed
        needed_imports = []
        if 'List[' in extracted_code:
            needed_imports.append('List')
        if 'Dict[' in extracted_code:
            needed_imports.append('Dict')
        if 'Tuple[' in extracted_code:
            needed_imports.append('Tuple')
        if 'Optional[' in extracted_code:
            needed_imports.append('Optional')
        if 'Union[' in extracted_code:
            needed_imports.append('Union')
        if 'Set[' in extracted_code:
            needed_imports.append('Set')
        if 'Any' in extracted_code and 'Any' not in needed_imports:
            needed_imports.append('Any')

        if needed_imports:
            typing_import = f"from typing import {', '.join(needed_imports)}\n"
            extracted_code = typing_import + extracted_code

    # Add common stdlib imports if used but not imported
    stdlib_imports = []
    if 'defaultdict' in extracted_code and 'from collections' not in extracted_code:
        stdlib_imports.append('from collections import defaultdict, deque, Counter')
    elif 'deque' in extracted_code and 'from collections' not in extracted_code:
        stdlib_imports.append('from collections import deque')
    elif 'Counter' in extracted_code and 'from collections' not in extracted_code:
        stdlib_imports.append('from collections import Counter')

    if 'heappush' in extracted_code or 'heappop' in extracted_code:
        if 'from heapq' not in extracted_code and 'import heapq' not in extracted_code:
            stdlib_imports.append('from heapq import heappush, heappop, heapify')

    if ' inf' in extracted_code or '(inf' in extracted_code or '[inf' in extracted_code:
        if 'inf = ' not in extracted_code and 'from math import inf' not in extracted_code:
            stdlib_imports.append('from math import inf')

    if stdlib_imports:
        extracted_code = '\n'.join(stdlib_imports) + '\n' + extracted_code
    
    # Combine with starter code if provided
    if starter_code:
        # Extract imports from starter code that might be needed
        starter_lines = starter_code.split('\n')
        starter_imports = [line for line in starter_lines
                          if line.strip().startswith(('import ', 'from '))]

        # Check if extracted code already has a class or function definition
        # If so, don't prepend starter code (model provided complete solution)
        has_class_def = 'class ' in extracted_code
        has_func_def = 'def ' in extracted_code

        # Only prepend starter code if the model output doesn't have its own structure
        if not has_class_def and not has_func_def:
            # Check if we need to add starter imports
            if starter_imports:
                extracted_lines = extracted_code.split('\n')
                existing_imports = [line for line in extracted_lines
                                  if line.strip().startswith(('import ', 'from '))]
                for imp in starter_imports:
                    if imp not in existing_imports:
                        extracted_code = imp + '\n' + extracted_code

            # Prepend starter code since model only output function body
            if starter_code.strip() not in extracted_code:
                combined = starter_code + "\n" + extracted_code
                return combined

    return extracted_code


def run_tests(code: str, test_code: str, entry_point: str = "candidate", timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Execute code and run tests in a subprocess with timeout.
    
    The test_code should contain a check(candidate) function that tests the solution.
    We'll call check(entry_point) where entry_point is the function name from the dataset.
    
    Args:
        code: The solution code to test
        test_code: The test code containing check(candidate) function
        entry_point: The name of the function to test (default: "candidate")
        timeout: Maximum execution time in seconds
        
    Returns:
        Tuple of (tests_passed: bool, error_message: Optional[str])
    """
    # Combine code and tests
    # The test_code contains check(candidate), so we need to call it with the entry_point function
    full_code = code + "\n\n" + test_code + f"\n\n# Run the tests\ncheck({entry_point})"
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_file = f.name
    
    try:
        # Run the code in a subprocess with timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(temp_file)
        )
        
        # Check if tests passed (exit code 0 typically means success)
        if result.returncode == 0:
            return True, None
        else:
            error_msg = result.stderr or result.stdout or ""
            return False, error_msg[:1500]  # Enough for NameError, AssertionError, traceback
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout} seconds"
    except Exception as e:
        return False, f"Execution error: {str(e)}"
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass

