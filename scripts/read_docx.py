"""Extract plain text from .docx resume files via raw XML — no python-docx dependency."""
import zipfile
import re

def extract_text(docx_path):
    with zipfile.ZipFile(docx_path) as z:
        with z.open('word/document.xml') as f:
            xml = f.read()
    # Simple text extraction via regex (good enough for review)
    text = re.sub(r'<[^>]+>', '\n', xml.decode('utf-8'))
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

files = [
    r"C:\Users\bibyd\Projects\personal\employment\Resume_DarrinBiby_OneDigital_OperationsAnalyst.docx",
    r"C:\Users\bibyd\Projects\personal\employment\Resume_DarrinBiby_Dudek_DataAnalyst.docx",
]

out_lines = []
for f in files:
    out_lines.append(f"\n\n=== {f.split(chr(92))[-1]} ===\n")
    try:
        out_lines.append(extract_text(f))
    except Exception as e:
        out_lines.append(f"ERROR: {e}")

with open("C:/Users/bibyd/Projects/dev/ApplyPilot/_resume_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print("DONE")
