import unittest

from clip_utils import chunk_transcript, normalize_clips


class NormalizeClipsTests(unittest.TestCase):
    def test_chunks_keep_all_segments_and_absolute_timestamps(self):
        segments = [
            {"start": 0, "text": "Opening"},
            {"start": 120, "text": "Middle"},
            {"start": 240, "text": "Ending"},
        ]
        chunks = chunk_transcript(segments, max_chars=1000)
        joined = "\\n".join(chunks)
        self.assertIn("[0.0s] Opening", joined)
        self.assertIn("[120.0s] Middle", joined)
        self.assertIn("[240.0s] Ending", joined)

    def test_chunks_respect_size_limit(self):
        segments = [{"start": i * 10, "text": "spoken words " * 30} for i in range(8)]
        chunks = chunk_transcript(segments, max_chars=1200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 1200 for chunk in chunks))

    def test_rejects_invalid_and_out_of_range_candidates(self):
        result = normalize_clips([
            {"start": -1, "end": 20, "score": 90},
            {"start": 10, "end": 10, "score": 80},
            {"start": 10, "end": 20, "score": "bad"},
            {"start": 15, "end": 29, "score": 75},
        ], duration=30, count=5, min_length=10, max_length=20)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["start"], 15)

    def test_keeps_highest_scoring_non_overlapping_clips(self):
        result = normalize_clips([
            {"start": 0, "end": 30, "score": 70},
            {"start": 10, "end": 35, "score": 95},
            {"start": 40, "end": 65, "score": 80},
        ], duration=90, count=5, min_length=15, max_length=40)
        self.assertEqual([item["score"] for item in result], [95, 80])

    def test_caps_count_and_sanitizes_model_fields(self):
        result = normalize_clips([
            {"start": 0, "end": 15, "score": 140, "title": "x" * 200},
            {"start": 20, "end": 35, "score": 50},
        ], duration=40, count=1, min_length=15, max_length=20)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["score"], 100)
        self.assertEqual(len(result[0]["title"]), 120)

    def test_rejects_nan_and_infinite_values(self):
        result = normalize_clips([
            {"start": float("nan"), "end": 20, "score": 90},
            {"start": 20, "end": float("inf"), "score": 90},
        ], duration=100, count=5, min_length=10, max_length=30)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
