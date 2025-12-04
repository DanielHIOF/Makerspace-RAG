# =============================================================================
# PDF Extraction with OCR - Simplified
# =============================================================================
"""
Simple PDF extraction: Try text first, if garbage -> OCR.
No overcomplicated fallback chains.
"""

def extract_pdf_fast(file_path, force_ocr=False):
    """
    PDF text extraction. Quick check for text, otherwise OCR.
    
    Args:
        file_path: Path to PDF file
        force_ocr: If True, skip text extraction and go straight to OCR
    
    Returns:
        (text, page_count) tuple
    """
    import fitz
    
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    if not force_ocr:
        # Quick check: count actual text in PDF
        text_length = sum(len(page.get_text("text").strip()) for page in doc)
        doc.close()
        
        if text_length < 500:
            print(f"  [PDF] Only {text_length} chars found - using OCR")
            force_ocr = True
        else:
            # Try text extraction
            try:
                import pymupdf4llm
                md_text = pymupdf4llm.to_markdown(file_path)
                if md_text and len(md_text.strip()) > 300:
                    print(f"  [PDF] Text extraction OK: {len(md_text)} chars from {total_pages} pages")
                    return md_text, total_pages
                else:
                    print(f"  [PDF] Text extraction got junk, using OCR")
                    force_ocr = True
            except Exception as e:
                print(f"  [PDF] Text extraction failed: {e}, using OCR")
                force_ocr = True
    else:
        doc.close()
    
    # OCR
    print(f"  [OCR] Starting OCR...")
    return extract_pdf_ocr(file_path), total_pages


def extract_pdf_ocr(file_path):
    """OCR extraction using EasyOCR."""
    import fitz
    import easyocr
    import numpy as np
    from PIL import Image
    import io
    
    print(f"  [OCR] First run downloads ~100MB models...")
    
    reader = easyocr.Reader(['no', 'en'], gpu=False)
    
    doc = fitz.open(file_path)
    all_text = []
    
    for page_num, page in enumerate(doc):
        print(f"  [OCR] Page {page_num + 1}/{len(doc)}...")
        
        # Render to image (2x zoom)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # OCR
        results = reader.readtext(np.array(img))
        
        # Filter by confidence
        page_text = [text for (_, text, conf) in results if conf > 0.3]
        
        if page_text:
            all_text.append(f"--- Side {page_num + 1} ---")
            all_text.append("\n".join(page_text))
    
    doc.close()
    return "\n\n".join(all_text)


# Test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text, pages = extract_pdf_fast(sys.argv[1])
        print(f"\n{'='*60}")
        print(f"Pages: {pages}, Chars: {len(text)}")
        print(f"{'='*60}")
        print(text[:3000])
