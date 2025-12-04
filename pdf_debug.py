"""Quick PDF diagnostic to understand why text extraction fails."""
import fitz
import sys

def diagnose_pdf(file_path):
    doc = fitz.open(file_path)
    
    print(f"\n{'='*60}")
    print(f"PDF DIAGNOSTIC: {file_path}")
    print(f"{'='*60}")
    print(f"Pages: {len(doc)}")
    
    for page_num, page in enumerate(doc):
        print(f"\n--- PAGE {page_num + 1} ---")
        
        # Get text in different ways
        text_simple = page.get_text("text")
        text_blocks = page.get_text("blocks")
        text_dict = page.get_text("dict")
        
        print(f"Simple text length: {len(text_simple)} chars")
        print(f"Text blocks: {len(text_blocks)}")
        
        # Check for images
        images = page.get_images()
        print(f"Images on page: {len(images)}")
        
        # Check text dict structure
        blocks = text_dict.get("blocks", [])
        text_blocks_count = sum(1 for b in blocks if b.get("type") == 0)
        image_blocks_count = sum(1 for b in blocks if b.get("type") == 1)
        print(f"Dict blocks - Text: {text_blocks_count}, Image: {image_blocks_count}")
        
        # Print first few text blocks with their content
        print("\nFirst 5 text blocks content:")
        for i, block in enumerate(blocks[:10]):
            if block.get("type") == 0:  # text block
                lines = block.get("lines", [])
                for line in lines[:3]:
                    spans = line.get("spans", [])
                    for span in spans:
                        text = span.get("text", "").strip()
                        if text:
                            print(f"  [{i}] '{text[:80]}{'...' if len(text) > 80 else ''}'")
        
        # Print actual raw text (first 500 chars)
        print(f"\nRaw text extract (first 500 chars):")
        print("-" * 40)
        print(text_simple[:500] if text_simple else "(empty)")
        print("-" * 40)
    
    doc.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_debug.py <pdf_file>")
        sys.exit(1)
    diagnose_pdf(sys.argv[1])
