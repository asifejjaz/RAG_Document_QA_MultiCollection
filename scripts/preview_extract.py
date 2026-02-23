#!/usr/bin/env python3
"""
Preview extracted text for inspection (page-aware for PDF).

Required CLI:
  python scripts/preview_extract.py --file "/data/hydrogen_books/book1.pdf" --pages 1-2
  python scripts/preview_extract.py --file "/data/biofuels_books/doc.docx" --head 2000
"""
import sys
from pathlib import Path

# Allow running from project root
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.preview import preview_file
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Show extracted text for inspection (page-aware for PDF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/preview_extract.py --file "/data/hydrogen_books/book1.pdf" --pages 1-2
  python scripts/preview_extract.py --file "/data/biofuels_books/doc.docx" --head 2000
  python scripts/preview_extract.py --file "/data/hydrogen_books/book1.pdf" --all
        """,
    )
    parser.add_argument("--file", type=str, required=True, help="Path to document (PDF, DOCX)")
    parser.add_argument("--pages", type=str, help='Page range for PDF (e.g. "1-2", "1,3,5")')
    parser.add_argument("--head", type=int, help="Number of characters to show for DOCX")
    parser.add_argument("--all", action="store_true", help="Show all content")
    args = parser.parse_args()
    if args.pages and args.head:
        parser.error("Cannot specify both --pages and --head")
    try:
        preview_file(
            file_path=args.file,
            pages=args.pages,
            head=args.head,
            show_all=args.all,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\nPreview interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Preview failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
