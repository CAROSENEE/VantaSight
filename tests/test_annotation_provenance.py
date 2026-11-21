import unittest

from vantasight.analytics import annotation_provenance


class AnnotationProvenanceTests(unittest.TestCase):
    def test_accepts_explicit_manual_review(self):
        result = annotation_provenance(
            {"reviewed_manually": True}, allow_unreviewed=False, source="labels.jsonl", line_number=1
        )
        self.assertEqual(result, "manually_reviewed")

    def test_rejects_provisional_record_by_default(self):
        with self.assertRaisesRegex(ValueError, "Unreviewed annotation"):
            annotation_provenance(
                {"review_note": "assistant_visual_provisional_user_review_needed"},
                allow_unreviewed=False,
                source="labels.jsonl",
                line_number=2,
            )

    def test_allows_unreviewed_only_as_exploration(self):
        result = annotation_provenance({}, allow_unreviewed=True, source="labels.jsonl", line_number=2)
        self.assertEqual(result, "unreviewed_exploratory")

    def test_provisional_note_overrides_manual_flag(self):
        with self.assertRaises(ValueError):
            annotation_provenance(
                {"reviewed_manually": True, "review_note": "provisional"},
                allow_unreviewed=False,
                source="labels.jsonl",
                line_number=2,
            )


if __name__ == "__main__":
    unittest.main()
