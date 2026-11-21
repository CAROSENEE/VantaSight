import argparse
import unittest

from vantasight.analytics import (
    annotation_provenance,
    parse_frame_indexes,
    validate_annotation_coverage,
)


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

    def test_parses_unique_expected_frame_indexes(self):
        self.assertEqual(parse_frame_indexes("0, 600,1200"), (0, 600, 1200))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_frame_indexes("0,0")

    def test_annotation_coverage_requires_exact_frame_set(self):
        records = [{"frame_index": 0}, {"frame_index": 600}]
        validate_annotation_coverage(records, [0, 600])
        with self.assertRaisesRegex(ValueError, r"missing frames: \[600\]"):
            validate_annotation_coverage(records[:1], [0, 600])
        with self.assertRaisesRegex(ValueError, r"unexpected frames: \[1200\]"):
            validate_annotation_coverage(records + [{"frame_index": 1200}], [0, 600])


if __name__ == "__main__":
    unittest.main()
