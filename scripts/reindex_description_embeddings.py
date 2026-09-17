"""Explicitly rebuild every persisted description embedding under one profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.database import init_db
from storage.embedding_index import reindex_description_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild files/table/column description embeddings and atomically "
            "record their configured model and encoding profile."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="documents encoded per model call (default: 128)",
    )
    args = parser.parse_args()
    init_db()
    report = reindex_description_embeddings(batch_size=args.batch_size)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
