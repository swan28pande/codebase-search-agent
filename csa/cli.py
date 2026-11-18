import argparse
import asyncio

from .graph import search
from .index import CodeIndex
from .llm import VLLMClient


def main():
    ap = argparse.ArgumentParser(description="Search a codebase with parallel agents.")
    ap.add_argument("root", help="path to the repository")
    ap.add_argument("question")
    ap.add_argument("--vllm", action="store_true", help="use a vLLM server (VLLM_BASE_URL, VLLM_MODEL)")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    index = CodeIndex(args.root)
    llm = VLLMClient() if args.vllm else None
    for r in asyncio.run(search(index, args.question, llm, k=args.k)):
        print(f"{r['ref']}  votes={r['votes']}/{r['agents']}  score={r['score']}")


if __name__ == "__main__":
    main()
