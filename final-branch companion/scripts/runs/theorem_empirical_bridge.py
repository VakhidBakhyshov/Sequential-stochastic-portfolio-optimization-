"""Run the synthetic theorem-to-data verification suite."""
from pathlib import Path
import argparse
from scripts.validation.theorem_empirical_bridge import run_synthetic_theorem_audit


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",default="results/theorem_empirical_bridge")
    p.add_argument("--seed",type=int,default=20260910)
    a=p.parse_args()
    out=Path(a.output_dir)
    summary=run_synthetic_theorem_audit(out,seed=a.seed)
    print(f"theorem audit written to {out.resolve()}")
    print(f"checks: {len(summary['checks'])}")

if __name__ == "__main__": main()
