"""
Phase 5 — Launch the Gradio interactive demo.

Usage
-----
    python scripts/run_phase5.py [--port PORT]

After verifying the demo works: run pytest tests/phase5/ to verify.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.demo.app import create_app


def main(port: int = 7860) -> None:
    print("=" * 60)
    print("PHASE 5 — Gradio Demo")
    print("=" * 60)
    print(f"Starting demo at http://localhost:{port}")
    app = create_app()
    app.launch(share=False, server_port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(args.port)
