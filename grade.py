import argparse
import json
from helixdesk.graders import run_all_graders
from rich.console import Console
from rich.table import Table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_all_graders(n_episodes=args.episodes)

    console = Console()
    table = Table(title="HelixDesk OpenEnv — Grader Results")
    table.add_column("Check", style="cyan")
    table.add_column("Passed", style="green")
    table.add_column("Score", justify="right")
    table.add_column("Detail")

    for check in results["checks"]:
        passed_str = "✓" if check["passed"] else "✗"
        table.add_row(check["check_name"], passed_str, f"{check['score']:.2f}", check["detail"])

    console.print(table)
    console.print(f"\n[bold]Total Score: {results['total_score']:.4f}[/bold]")
    console.print(f"All Checks Passed: {results['passed_all']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\nResults written to {args.output}")

if __name__ == "__main__":
    main()
