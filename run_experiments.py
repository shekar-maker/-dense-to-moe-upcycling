"""Trains every run in moe/experiments.py in order (finished runs are skipped).

    python run_experiments.py                 # all runs
    python run_experiments.py moe_copy ...    # only the named runs
"""
import sys

from moe.data import prepare
from moe.experiments import ALL, run_all

if __name__ == "__main__":
    print(prepare("data"))
    run_all(sys.argv[1:] or ALL)
