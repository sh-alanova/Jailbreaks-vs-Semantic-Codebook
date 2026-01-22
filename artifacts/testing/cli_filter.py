import argparse
import sys

from semantic_filter import is_harmful


def main():
    parser = argparse.ArgumentParser(
        description="Cross-lingual semantic jailbreak filter CLI — checks if a prompt is harmful."
    )
    parser.add_argument(
        "--prompt",
        nargs="?",
        help="Input prompt to check (if not provided, reads from stdin)",
    )
    parser.add_argument("--stdin", action="store_true", help="Read prompt from stdin")

    args = parser.parse_args()

    if args.stdin:
        prompt = sys.stdin.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        print("Error: Please provide a prompt or use --stdin.", file=sys.stderr)
        sys.exit(1)

    if not prompt:
        print("Error: Empty prompt.", file=sys.stderr)
        sys.exit(1)

    try:
        harmful = is_harmful(prompt)
        status = "BLOCKED" if harmful else "ALLOWED"
        print(status)
        sys.exit(1 if harmful else 0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
