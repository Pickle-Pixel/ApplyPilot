"""F11: cover-letter PDFs render the body (not garbage) and escape content."""
from applypilot.scoring.pdf import _letter_html


def test_letter_html_keeps_all_paragraphs():
    letter = "Dear Hiring Manager,\n\nI built systems with List<String>.\n\nSincerely,\nJane"
    html = _letter_html(letter, "Jane Doe")
    assert "Dear Hiring Manager," in html
    assert "Sincerely," in html
    # Middle paragraph survives.
    assert "I built systems" in html


def test_letter_html_escapes_dangerous_content():
    html = _letter_html("I used List<String> and <script>alert(1)</script>.", "Jane")
    assert "List&lt;String&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_name_appears_once_as_header():
    html = _letter_html("Body text here.", "Jane Doe")
    assert html.count('class="name">Jane Doe') == 1
