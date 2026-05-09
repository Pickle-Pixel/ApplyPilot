"""Inspect MASTER_RESUME.docx paragraph structure to plan surgical edits."""
from docx import Document
from pathlib import Path
p = Path(r"C:\Users\bibyd\.applypilot\resume-overhaul\MASTER_RESUME.docx")
doc = Document(p)
for i, para in enumerate(doc.paragraphs[:20]):
    style = (para.style.name if para.style else "")
    text = para.text
    print(f"[{i}] style={style!r} text={text[:120]!r}")
