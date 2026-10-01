"""HumanEval + MBPP pass@1 runner for CodeForge-250M (greedy, KV-cache decoding).

Runs both benchmarks via evaluation/codeforge_eval.py with the MBPP prompt protocol used for the
Stage A report (CF_MB_STYLE=named). Needs HF_TOKEN (env or Kaggle secret) and internet.
Stage A result (HF rev 2542b768, step 2898): HumanEval 7/164 = 4.27%, MBPP 13/500 = 2.60%.
"""
import os, runpy
os.environ.setdefault("CF_MB_STYLE", "named")
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "codeforge_eval.py"), run_name="__main__")
