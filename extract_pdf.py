import fitz  # PyMuPDF
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import cv2
import numpy as np
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

PDF_PATH = "data/ltimindtree_annual_report.pdf"
TEXT_CHUNK_PATH = "output/text_chunks.json"
IMAGE_OUTPUT_DIR = "output/figures"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)


def is_magazine_format(pdf_path, threshold_ratio=1.2, ratio_cutoff=0.5):
    doc = fitz.open(pdf_path)
    wide_pages = 0

    for i in range(len(doc)):
        page = doc[i]
        width, height = page.rect.width, page.rect.height
        aspect_ratio = width / height
        if aspect_ratio > threshold_ratio:
            wide_pages += 1

    return wide_pages > len(doc) * ratio_cutoff


def process_page(page_num, is_spread, pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    results = []

    # Render page as image
    pix = page.get_pixmap(dpi=200)  # faster
    img_path = f"{IMAGE_OUTPUT_DIR}/page_{page_num + 1}.png"
    pix.save(img_path)

    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print(f"[PAGE {page_num + 1}] ❌ Image failed to load.")
        return results

    h, w = img_cv.shape[:2]

    if is_spread:
        left = img_cv[:, :w // 2]
        right = img_cv[:, w // 2:]

        left_text = pytesseract.image_to_string(left, config="--psm 3")
        right_text = pytesseract.image_to_string(right, config="--psm 3")

        if left_text.strip():
            print(f"[PAGE {page_num + 1} LEFT] ✅")
            results.append({
                "page": page_num + 1,
                "text": left_text.strip(),
                "source": f"{img_path} (left)"
            })

        if right_text.strip():
            print(f"[PAGE {page_num + 1} RIGHT] ✅")
            results.append({
                "page": page_num + 1,
                "text": right_text.strip(),
                "source": f"{img_path} (right)"
            })

        if not left_text.strip() and not right_text.strip():
            print(f"[PAGE {page_num + 1}] ⚠️ No OCR text found in halves.")

    else:
        # Standard full-page text or fallback OCR
        text = page.get_text()
        if text.strip():
            print(f"[PAGE {page_num + 1}] ✅ Text extracted normally.")
            results.append({"page": page_num + 1, "text": text.strip()})
        else:
            ocr_text = pytesseract.image_to_string(img_cv)
            if ocr_text.strip():
                print(f"[PAGE {page_num + 1}] ✅ OCR fallback used.")
                results.append({
                    "page": page_num + 1,
                    "text": ocr_text.strip(),
                    "source": img_path
                })

    # Embedded image OCR (optional - slower)
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_path = f"{IMAGE_OUTPUT_DIR}/page{page_num + 1}_img{img_index + 1}.{image_ext}"

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        img_cv = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_cv is not None:
            ocr_text = pytesseract.image_to_string(img_cv)
            if ocr_text.strip():
                print(f"[PAGE {page_num + 1} | IMG {img_index + 1}] ✅ OCR image extracted.")
                results.append({
                    "page": page_num + 1,
                    "text": ocr_text.strip(),
                    "source": image_path
                })

    return results


def extract_text_and_images(pdf_path):
    print("🚀 Starting parallel OCR extraction...")
    doc = fitz.open(pdf_path)
    is_spread = is_magazine_format(pdf_path)
    print(f"📄 Detected layout: {'MAGAZINE/SPREAD' if is_spread else 'SINGLE-PAGE'}")

    all_blocks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_page, i, is_spread, pdf_path) for i in range(len(doc))]
        for future in as_completed(futures):
            blocks = future.result()
            if blocks:
                all_blocks.extend(blocks)

    print(f"✅ Extracted {len(all_blocks)} content chunks.")
    return all_blocks


if __name__ == "__main__":
    PDF_PATH = "data/ltimindtree_annual_report.pdf"
    OUTPUT_PATH = "output/text_chunks.json"

    chunks = extract_text_and_images(PDF_PATH)

    # Sort by page number, then left/right/embedded (alphabetically by 'source')
    chunks.sort(key=lambda c: (c['page'], c.get('source', '')))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Sorted and saved {len(chunks)} chunks.")


    print(f"📝 Saved to {OUTPUT_PATH}")