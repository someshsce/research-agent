"""CLI runner: `python run_cli.py "Razorpay"` — useful for quick tests."""
from __future__ import annotations

import json
import sys

from dotenv import load_dotenv

load_dotenv()

from agent.graph import run_agent  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python run_cli.py "<query>"')
        print('  e.g.: python run_cli.py "Razorpay"')
        print('        python run_cli.py "React vs Vue"')
        print('        python run_cli.py "carbon capture technology"')
        return 1

    query = " ".join(sys.argv[1:])
    print(f"\n🔍 Researching: {query}\n")
    final_state = run_agent(query)

    print("\n" + "=" * 60)
    print("AGENT LOG")
    print("=" * 60)
    for line in final_state.get("logs", []):
        print(line)

    report = final_state.get("final_report")
    if report is None:
        print("\n❌ No final report produced.")
        return 2

    print("\n" + "=" * 60)
    print(f"STRUCTURED REPORT ({report.report_type})")
    print("=" * 60)
    print(json.dumps(report.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
