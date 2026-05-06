# Ingestion module: turns a file on disk into raw text.(Text extraction for files)

from pathlib import Path
from pypdf import PdfReader
from docx import Document

def extract_from_pdf(file_path: Path) -> str:
    #Extract text from a PDF, page by page.
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # some pages can return None (e.g. images)
            text += page_text + "\n"
    return text


def extract_from_docx(file_path: Path) -> str:
    #Extract text from a Word document.
    doc = Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def extract_from_txt(file_path: Path) -> str:
    #Read plain text file.
    return file_path.read_text(encoding="utf-8", errors="ignore")


def extract_text(file_path: Path) -> str:
    #Dispatch to the right extractor based on file extension.
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_from_pdf(file_path)
    elif suffix == ".docx":
        return extract_from_docx(file_path)
    elif suffix == ".txt":
        return extract_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")