"""도안 파일 읽기: PDF 텍스트 추출, 텍스트가 없는 스캔본/이미지는 페이지 이미지로 변환."""
from __future__ import annotations

import base64


def read_upload(filename: str, data: bytes, max_pages: int = 6) -> dict:
    """returns {"text": str, "images": [data_uri,...], "kind": "pdf_text|pdf_scan|image|text"}"""
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            import pymupdf as fitz  # PyMuPDF 최신
        except ImportError:
            import fitz  # PyMuPDF 구버전

        doc = fitz.open(stream=data, filetype="pdf")
        pages = [doc[i] for i in range(min(len(doc), max_pages))]
        text = "\n".join(p.get_text("text") for p in pages).strip()
        if len(text) >= 80:
            return {"text": text, "images": [], "kind": "pdf_text", "pages": len(doc)}
        images = []
        for p in pages[:4]:
            pix = p.get_pixmap(dpi=150)
            images.append("data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode())
        return {"text": text, "images": images, "kind": "pdf_scan", "pages": len(doc)}
    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        mime = "image/png" if name.endswith(".png") else "image/webp" if name.endswith(".webp") else "image/jpeg"
        return {"text": "", "images": [f"data:{mime};base64," + base64.b64encode(data).decode()], "kind": "image", "pages": 1}
    for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return {"text": data.decode(enc), "images": [], "kind": "text", "pages": 1}
        except UnicodeDecodeError:
            continue
    return {"text": "", "images": [], "kind": "text", "pages": 1}
