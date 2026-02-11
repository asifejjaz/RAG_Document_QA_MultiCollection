#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF with page awareness
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        List of page dictionaries with text and metadata
    """
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
        pages = []
        for doc in docs:
            pages.append({
                'page_number': doc.metadata.get('page', 0) + 1,  # 1-indexed
                'text': doc.page_content,
                'char_count': len(doc.page_content),
                'word_count': len(doc.page_content.split()),
                'metadata': doc.metadata
            })
        
        return pages
        
    except Exception as e:
        logger.error(f"Failed to extract PDF: {e}")
        return []


def extract_text_from_docx(file_path: str) -> Dict[str, Any]:
    """
    Extract text from DOCX file
    
    Args:
        file_path: Path to DOCX file
        
    Returns:
        Dictionary with text and metadata
    """
    try:
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        
        # Combine all sections
        full_text = "\n\n".join([doc.page_content for doc in docs])
        
        return {
            'text': full_text,
            'char_count': len(full_text),
            'word_count': len(full_text.split()),
            'sections': len(docs)
        }
        
    except Exception as e:
        logger.error(f"Failed to extract DOCX: {e}")
        return {}


# ============================================================================
# PAGE RANGE PARSING
# ============================================================================

def parse_page_range(page_spec: str, total_pages: int) -> List[int]:
    """
    Parse page specification into list of page numbers
    
    Args:
        page_spec: Page specification (e.g., "1", "1-5", "1,3,5")
        total_pages: Total number of pages available
        
    Returns:
        List of page numbers (1-indexed)
    """
    pages = []
    
    # Handle comma-separated values
    parts = page_spec.split(',')
    
    for part in parts:
        part = part.strip()
        
        # Range (e.g., "1-5")
        if '-' in part:
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())
                
                # Validate range
                if start < 1 or end > total_pages or start > end:
                    logger.warning(f"Invalid range {part}, total pages: {total_pages}")
                    continue
                
                pages.extend(range(start, end + 1))
                
            except ValueError:
                logger.warning(f"Invalid range format: {part}")
                continue
        
        # Single page
        else:
            try:
                page_num = int(part)
                
                if page_num < 1 or page_num > total_pages:
                    logger.warning(f"Page {page_num} out of range (1-{total_pages})")
                    continue
                
                pages.append(page_num)
                
            except ValueError:
                logger.warning(f"Invalid page number: {part}")
                continue
    
    # Remove duplicates and sort
    pages = sorted(list(set(pages)))
    
    return pages


# ============================================================================
# PREVIEW FORMATTING
# ============================================================================

def format_pdf_preview(
    pages: List[Dict[str, Any]],
    page_numbers: Optional[List[int]] = None,
    show_all: bool = False
) -> str:
    """
    Format PDF pages for preview
    
    Args:
        pages: List of page dictionaries
        page_numbers: Specific page numbers to show (None for all)
        show_all: Show all pages
        
    Returns:
        Formatted preview string
    """
    lines = []
    
    # Header
    total_pages = len(pages)
    lines.append("\n" + "="*80)
    lines.append("PDF TEXT EXTRACTION PREVIEW")
    lines.append("="*80)
    lines.append(f"Total Pages: {total_pages}")
    
    # Determine which pages to show
    if show_all:
        pages_to_show = list(range(1, total_pages + 1))
        lines.append("Showing: All pages")
    elif page_numbers:
        pages_to_show = page_numbers
        lines.append(f"Showing: Pages {', '.join(map(str, page_numbers))}")
    else:
        pages_to_show = [1]  # Default to first page
        lines.append("Showing: Page 1 (use --pages or --all for more)")
    
    lines.append("="*80 + "\n")
    
    # Show pages
    for page_num in pages_to_show:
        if page_num < 1 or page_num > total_pages:
            continue
        
        page = pages[page_num - 1]  # Convert to 0-indexed
        
        lines.append(f"\n{'─'*80}")
        lines.append(f"PAGE {page_num}")
        lines.append(f"{'─'*80}")
        lines.append(f"Characters: {page['char_count']:,}")
        lines.append(f"Words: {page['word_count']:,}")
        lines.append(f"{'─'*80}\n")
        
        lines.append(page['text'])
        lines.append("\n")
    
    # Footer
    lines.append("="*80)
    lines.append(f"Preview Complete - Showed {len(pages_to_show)} page(s)")
    lines.append("="*80 + "\n")
    
    return "\n".join(lines)


def format_docx_preview(
    doc_data: Dict[str, Any],
    head_chars: Optional[int] = None
) -> str:
    """
    Format DOCX content for preview
    
    Args:
        doc_data: Document data dictionary
        head_chars: Number of characters to show (None for all)
        
    Returns:
        Formatted preview string
    """
    lines = []
    
    # Header
    lines.append("\n" + "="*80)
    lines.append("DOCX TEXT EXTRACTION PREVIEW")
    lines.append("="*80)
    lines.append(f"Total Characters: {doc_data['char_count']:,}")
    lines.append(f"Total Words: {doc_data['word_count']:,}")
    lines.append(f"Sections: {doc_data['sections']}")
    
    # Determine what to show
    text = doc_data['text']
    
    if head_chars and head_chars < len(text):
        text_to_show = text[:head_chars]
        truncated = True
        lines.append(f"Showing: First {head_chars:,} characters")
    else:
        text_to_show = text
        truncated = False
        lines.append("Showing: Full content")
    
    lines.append("="*80 + "\n")
    
    # Content
    lines.append(text_to_show)
    
    if truncated:
        lines.append("\n\n[... truncated ...]")
    
    # Footer
    lines.append("\n" + "="*80)
    
    if truncated:
        remaining = len(text) - head_chars
        lines.append(f"Remaining: {remaining:,} characters ({remaining / len(text) * 100:.1f}%)")
    else:
        lines.append("Preview Complete - Showed full content")
    
    lines.append("="*80 + "\n")
    
    return "\n".join(lines)


# ============================================================================
# PREVIEW EXECUTION
# ============================================================================

def preview_file(
    file_path: str,
    pages: Optional[str] = None,
    head: Optional[int] = None,
    show_all: bool = False
) -> None:
    """
    Preview extracted text from file
    
    Args:
        file_path: Path to file
        pages: Page specification for PDFs (e.g., "1-5", "1,3,5")
        head: Number of characters to show for DOCX
        show_all: Show all content
    """
    file_path = Path(file_path)
    
    # Validate file
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    extension = file_path.suffix.lower()
    
    # Handle PDF
    if extension == '.pdf':
        logger.info(f"Extracting text from PDF: {file_path.name}")
        
        page_data = extract_text_from_pdf(str(file_path))
        
        if not page_data:
            logger.error("Failed to extract text from PDF")
            sys.exit(1)
        
        # Parse page range
        page_numbers = None
        if pages and not show_all:
            page_numbers = parse_page_range(pages, len(page_data))
            
            if not page_numbers:
                logger.error("No valid pages specified")
                sys.exit(1)
        
        # Format and print preview
        preview = format_pdf_preview(page_data, page_numbers, show_all)
        print(preview)
    
    # Handle DOCX
    elif extension in ['.docx', '.doc']:
        logger.info(f"Extracting text from DOCX: {file_path.name}")
        
        doc_data = extract_text_from_docx(str(file_path))
        
        if not doc_data:
            logger.error("Failed to extract text from DOCX")
            sys.exit(1)
        
        # Format and print preview
        preview = format_docx_preview(doc_data, head)
        print(preview)
    
    else:
        logger.error(f"Unsupported file type: {extension}")
        logger.error("Supported types: .pdf, .docx, .doc")
        sys.exit(1)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Preview extracted text from documents (page-aware for PDF)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PDF: Show pages 1-2
  python scripts/preview_extract.py --file "/data/hydrogen_books/book1.pdf" --pages 1-2
  
  # PDF: Show specific pages
  python scripts/preview_extract.py --file "/data/hydrogen_books/book1.pdf" --pages 1,5,10
  
  # PDF: Show all pages
  python scripts/preview_extract.py --file "/data/hydrogen_books/book1.pdf" --all
  
  # DOCX: Show first 2000 characters
  python scripts/preview_extract.py --file "/data/biofuels_books/doc.docx" --head 2000
  
  # DOCX: Show all content
  python scripts/preview_extract.py --file "/data/biofuels_books/doc.docx" --all
        """
    )
    
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Path to document file (PDF, DOCX)'
    )
    
    parser.add_argument(
        '--pages',
        type=str,
        help='Page range for PDF (e.g., "1-2", "1,3,5", "1-5,10")'
    )
    
    parser.add_argument(
        '--head',
        type=int,
        help='Number of characters to show for DOCX files'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Show all content (all pages for PDF, full text for DOCX)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.pages and args.head:
        parser.error("Cannot specify both --pages and --head")
    
    try:
        preview_file(
            file_path=args.file,
            pages=args.pages,
            head=args.head,
            show_all=args.all
        )
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("\nPreview interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Preview failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()