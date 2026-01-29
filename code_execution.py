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
    
    # Combine with starter code if provided
    if starter_code:
        # If starter code has a function definition, try to merge intelligently
        if starter_code.strip() and not completion.strip().startswith(starter_code.strip()):
            # Simple concatenation - might need refinement based on actual data
            combined = starter_code + "\n" + completion
            return combined
    
    return completion.strip()


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
            error_msg = result.stderr or result.stdout
            return False, error_msg[:500]  # Limit error message length
            
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

