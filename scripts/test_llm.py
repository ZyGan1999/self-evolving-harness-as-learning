"""One-shot LLM connectivity test (for official APIs or relays like packyapi).

Usage:
  conda run -n appworld-p python scripts/test_llm.py anthropic:claude-haiku-4-5-20251001
  conda run -n appworld-p python scripts/test_llm.py openai:claude-haiku-4-5-20251001
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.llm import build_llm  # noqa: E402


def main() -> None:
    spec = sys.argv[1] if len(sys.argv) > 1 else "anthropic:claude-haiku-4-5-20251001"
    llm = build_llm(spec)
    reply = llm.generate(
        system="You are a test probe. Answer in one short line.",
        messages=[{"role": "user", "content": "Reply with exactly: CONNECTIVITY OK"}],
        max_tokens=32, temperature=0.0,
    )
    print(f"spec:  {spec}")
    print(f"reply: {reply.strip()}")
    print(f"usage: {llm.usage}")


if __name__ == "__main__":
    main()
