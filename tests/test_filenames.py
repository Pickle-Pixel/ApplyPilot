"""F9: per-job artifact filenames must not collide; uploads are per-worker."""
import applypilot.config as config
import applypilot.apply.prompt as prompt_mod
from applypilot.scoring.tailor import make_filename_prefix


def test_same_title_site_different_url_distinct_prefix():
    a = make_filename_prefix({"title": "Software Engineer", "site": "linkedin",
                              "url": "https://example.com/a"})
    b = make_filename_prefix({"title": "Software Engineer", "site": "linkedin",
                              "url": "https://example.com/b"})
    assert a != b


def test_same_job_stable_prefix():
    job = {"title": "Software Engineer", "site": "linkedin", "url": "https://example.com/a"}
    assert make_filename_prefix(job) == make_filename_prefix(job)


def _fixture_profile():
    return {
        "personal": {"full_name": "Jane Doe", "email": "j@example.com",
                     "phone": "5551234567", "city": "Austin"},
        "work_authorization": {},
        "compensation": {"salary_expectation": "100000"},
        "experience": {}, "availability": {}, "eeo": {}, "skills_boundary": {},
    }


def test_upload_dir_is_per_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load_profile", _fixture_profile)
    monkeypatch.setattr(config, "load_search_config", lambda: {})
    monkeypatch.setattr(config, "APPLY_WORKER_DIR", tmp_path)
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    job = {"url": "https://example.com/j", "title": "Engineer", "site": "linkedin",
           "application_url": None, "fit_score": 8, "tailored_resume_path": str(tmp_path / "x.txt")}
    out = prompt_mod.build_prompt(job=job, tailored_resume="r", worker_id=2)
    assert "worker-2" in out
    assert (tmp_path / "worker-2" / "current" / "Jane_Doe_Resume.pdf").exists()
