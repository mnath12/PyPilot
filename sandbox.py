# sandbox.py  ────────────────────────────────────────────────────────────────
"""
Tiny code-evaluation sandbox for PyPilot RL.

Usage
-----
>>> from sandbox import run_tests
>>> user_code = open("generated_solution.py").read()
>>> tests     = leet_ds[4]['test']              # or any str containing asserts
>>> result = run_tests(user_code,
...                    tests_src=tests,
...                    candidate_expr="Solution().longestPalindrome",
...                    timeout=2)               # seconds
>>> print(result)
# {'compile': True, 'passed': True, 'error': None, 'traceback': None}
"""

import ast
import multiprocessing as mp
import textwrap
import traceback
import types
from typing import Dict


class SandboxError(Exception):
    """Custom wrapper so caller can differentiate sandbox issues."""
    pass


# ────────────────────────────────────────────────────────────────────────────
# 1. Syntax check
# ────────────────────────────────────────────────────────────────────────────
def _compile_check(src: str) -> None:
    """Raises SandboxError on SyntaxError."""
    try:
        ast.parse(src, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"Syntax error:\n{exc}") from exc


# ────────────────────────────────────────────────────────────────────────────
# 2. Load user code as an in-memory module
# ────────────────────────────────────────────────────────────────────────────
def _load_module(src: str, module_name: str = "user_module") -> types.ModuleType:
    """
    Execs `src` in a fresh module namespace and returns that module.
    NB: We leave built-ins untouched so things like `range`, `len` work.
    """
    mod = types.ModuleType(module_name)
    exec(src, mod.__dict__)
    return mod


# ────────────────────────────────────────────────────────────────────────────
# 3. Public helper
# ────────────────────────────────────────────────────────────────────────────
def run_tests(
    user_src: str,
    tests_src: str,
    candidate_expr: str,
    timeout: int = 2,
) -> Dict[str, object]:
    """
    Compile-checks, then evaluates tests in a subprocess.

    Parameters
    ----------
    user_src       : str   • Raw Python code produced by the model.
    tests_src      : str   • The LeetCode-style test string (asserts inside a
                            function `check(candidate)`).
    candidate_expr : str   • Python expression that yields the *callable* the
                            tests expect (e.g. "candidate",
                            "Solution().longestPalindrome").
    timeout        : int   • Seconds before killing runaway code.

    Returns
    -------
    dict with keys
        compile   : bool  • False ⇒ syntax error.
        passed    : bool  • True  ⇒ all asserts passed.
        error     : str|None
        traceback : str|None
    """
    out = {"compile": True, "passed": False, "error": None, "traceback": None}

    # 3-A  Syntax
    try:
        _compile_check(user_src)
    except SandboxError as e:
        out.update({"compile": False, "error": str(e)})
        return out

    # 3-B  Import user code
    mod = _load_module(user_src)
    ns = mod.__dict__.copy()          # namespace in which tests run

    try:
        candidate_fn = eval(candidate_expr, ns)
    except Exception as e:
        out.update({"error": f"Unable to resolve candidate ({candidate_expr}): {e}"})
        return out

    ns["candidate"] = candidate_fn    # what the tests will call

    # Wrap test string so we can invoke it easily
    wrapped = "def _run_tests():\n" + textwrap.indent(tests_src, "    ")

    try:
        exec(wrapped, ns)
    except Exception as e:
        out.update({"error": f"Invalid test string: {e}"})
        return out

    # 3-C  Execute tests in *another* process (protects kernel & enforces timeout)
    def _worker(q):
        try:
            ns["_run_tests"]()
            q.put(None)               # success marker
        except Exception:
            q.put(traceback.format_exc())

    queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(queue,))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():               # Guard against infinite loops / OOM
        proc.terminate()
        out.update({"error": "Time-out (> {} s)".format(timeout)})
        return out

    exc = queue.get()
    if exc is None:
        out["passed"] = True
    else:
        out.update({"error": "Test failure", "traceback": exc})

    return out
