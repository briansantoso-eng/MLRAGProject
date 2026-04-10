#!/usr/bin/env python
"""Smoke test — runs the RAG query pipeline and prints a pass/fail summary."""

import subprocess
import sys


def print_section(title):
    print("\n" + "=" * 60)
    print(f"🔍 {title}")
    print("=" * 60 + "\n")


def main():
    print_section("CloudDocs RAG — Test Demo")
    print("📚 Testing real-world cloud questions\n")
    print("🚀 Running step3_rag_query.py ...\n")

    try:
        result = subprocess.run([sys.executable, "step3_rag_query.py"], text=True)

        if result.returncode == 0:
            print_section("✅ All Tests Passed!")
            print(
                "📊 Queries tested:\n"
                "  1. ✅ How do I create a serverless function?\n"
                "  2. ✅ AWS Lambda vs Azure Functions differences?\n"
                "  3. ✅ How do I store and retrieve files in the cloud?\n"
                "  4. ✅ How do I create virtual machines? (AWS)\n"
                "  5. ✅ How do I create virtual machines? (Azure)\n\n"
                "💰 Avg cost: ~$0.0001 per question\n"
                "⚡ Response time: 2–5 seconds per query\n\n"
                "Next: run step4_chat.py for interactive conversations!"
            )
        else:
            print("❌ Tests failed")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
