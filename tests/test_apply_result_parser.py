import unittest

from applypilot.apply.launcher import parse_apply_result


class ApplyResultParserTests(unittest.TestCase):
    def test_exact_result_markers_win(self):
        self.assertEqual(parse_apply_result("RESULT:APPLIED"), "applied")
        self.assertEqual(parse_apply_result("RESULT:EXPIRED"), "expired")
        self.assertEqual(parse_apply_result("RESULT:DRY_RUN_READY"), "dry_run_ready")

    def test_submission_confirmation_without_result_line_counts_applied(self):
        output = """
        Application Successfully Submitted
        Confirmation URL:
        https://perstechinc.isolvedhire.com/apply/thanks.php?orig_job_app=1752348
        Thank you for completing the additional information for your application.
        """
        self.assertEqual(parse_apply_result(output), "applied")

    def test_expired_page_without_result_line_counts_expired(self):
        output = "The page you are looking for doesn't exist. This job posting is no longer available."
        self.assertEqual(parse_apply_result(output), "expired")

    def test_browser_backend_loss_is_distinct_from_no_result(self):
        output = "The browser backend at localhost:9222 is being refused."
        self.assertEqual(parse_apply_result(output), "failed:browser_backend_lost")

    def test_unknown_output_remains_no_result(self):
        self.assertEqual(parse_apply_result("I navigated around but stopped."), "failed:no_result_line")

    def test_none_output_treated_as_no_result(self):
        self.assertEqual(parse_apply_result(""), "failed:no_result_line")

    def test_login_issue_markers_keep_auth_context(self):
        self.assertEqual(parse_apply_result("Could not sign in because of invalid credentials."), "login_issue")
        self.assertEqual(parse_apply_result("Please log in to continue with the application."), "login_issue")
        self.assertEqual(parse_apply_result("Please login to your account details page to proceed."), "login_issue")

    def test_unrelated_login_words_remain_no_result(self):
        self.assertEqual(parse_apply_result("We can meet to sign in the morning after review."), "failed:no_result_line")
        self.assertEqual(parse_apply_result("Login attempts are tracked by the team."), "failed:no_result_line")

    def test_stale_transcript_with_noise_does_not_hide_no_result(self):
        output = """
        {"type":"assistant","message":{"content":[{"type":"text","text":"{\"json\": true}"}]}}
        no result marker intentionally missing
        """
        self.assertEqual(parse_apply_result(output), "failed:no_result_line")

    def test_result_marker_without_reason_defaults_to_unknown(self):
        self.assertEqual(parse_apply_result("RESULT:FAILED"), "failed:unknown")

    def test_result_marker_with_unknown_text_after_confirmation_stays_failed(self):
        output = (
            "Application submitted? maybe not.\n"
            "RESULT:FAILED: manual review recommended by policy checks."
        )
        self.assertEqual(parse_apply_result(output), "failed:manual_review_recommended_by_policy_checks")

    def test_failed_result_reason_with_spaces_normalizes(self):
        output = "RESULT:FAILED: login required"
        self.assertEqual(parse_apply_result(output), "failed:login_required")

    def test_browser_backend_lost_is_no_retry_terminal_failure_reason(self):
        output = "The browser backend at localhost:9222 is being refused."
        self.assertEqual(parse_apply_result(output), "failed:browser_backend_lost")

    def test_lowercase_manual_review_hint_marked_as_failure(self):
        output = "This flow required manual review and will not be submitted."
        self.assertEqual(parse_apply_result(output), "failed:manual_review_required")


if __name__ == "__main__":
    unittest.main()
