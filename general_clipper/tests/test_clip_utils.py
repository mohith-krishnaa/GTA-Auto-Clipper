import unittest

from clip_utils import chunk_transcript, crop_x_expression, group_caption_words, normalize_clips
from visual_utils import detect_gta_hud_cues, rank_gta_event_windows, rank_visual_windows


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

    def test_crop_expression_tracks_face_centers(self):
        expression = crop_x_expression([(0, 500), (2, 900)], 400, 1200)
        self.assertIn("if(lt(t", expression)
        self.assertIn("300.00", expression)
        self.assertIn("700.00", expression)

    def test_crop_expression_clamps_to_frame_bounds(self):
        expression = crop_x_expression([(0, -100)], 400, 1200)
        self.assertEqual(expression, "0.00")

    def test_caption_words_are_clipped_and_grouped(self):
        words = [
            {"start": 0.0, "end": 0.2, "text": "Hello"},
            {"start": 0.21, "end": 0.4, "text": "there"},
            {"start": 0.5, "end": 0.7, "text": "friend"},
            {"start": 2.0, "end": 2.2, "text": "Next"},
            {"start": 2.21, "end": 2.4, "text": "part"},
        ]
        groups = group_caption_words(words, 0.1, 2.3, max_words=3, max_gap=0.55)
        self.assertEqual([[w["text"] for w in group] for group in groups],
                         [["Hello", "there", "friend"], ["Next", "part"]])
        self.assertEqual(groups[0][0]["start"], 0.1)
        self.assertEqual(groups[-1][-1]["end"], 2.3)

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


class VisualWindowTests(unittest.TestCase):
    def test_ranks_high_activity_windows_without_overlap(self):
        samples = [{"timestamp": i, "motion": 0.02, "cut": 0.0} for i in range(100)]
        samples += [{"timestamp": i, "motion": 0.8, "cut": 0.3} for i in range(20, 35)]
        samples += [{"timestamp": i, "motion": 0.7, "cut": 0.2} for i in range(65, 80)]
        clips = rank_visual_windows(samples, 100, 3, 15, 25, step=5)
        self.assertEqual(len(clips), 2)
        self.assertTrue(all(clip["score"] > 0 for clip in clips))
        self.assertTrue(all(clips[i]["end"] <= clips[i + 1]["start"] for i in range(len(clips) - 1)))

    def test_static_footage_returns_no_false_candidates(self):
        samples = [{"timestamp": i, "motion": 0.0, "cut": 0.0} for i in range(60)]
        self.assertEqual(rank_visual_windows(samples, 60, 5, 15, 25), [])

    def test_ignores_invalid_visual_samples(self):
        samples = [{"timestamp": float("nan"), "motion": 1}, {"timestamp": 10, "motion": 0.9, "cut": 0.2}]
        clips = rank_visual_windows(samples, 40, 2, 15, 20)
        self.assertTrue(all(0 <= clip["start"] < clip["end"] <= 40 for clip in clips))



class GtaHudEventTests(unittest.TestCase):
    def test_detects_known_gta_hud_cues_and_ignores_generic_text(self):
        hits = detect_gta_hud_cues([
            {"timestamp": 10, "text": "MISSION PASSED"},
            {"timestamp": 20, "text": "WASTED"},
            {"timestamp": 30, "text": "this was a crazy mission"},
            {"timestamp": 40, "text": "NEW RECORD"},
        ])
        self.assertEqual([hit["cue"] for hit in hits], ["MISSION PASSED", "WASTED", "NEW RECORD"])

    def test_ranks_event_clips_with_lead_in_and_no_overlap(self):
        hits = [
            {"timestamp": 30, "cue": "WASTED", "weight": 1.0},
            {"timestamp": 32, "cue": "WASTED", "weight": 1.0},
            {"timestamp": 100, "cue": "MISSION PASSED", "weight": 1.0},
        ]
        clips = rank_gta_event_windows(hits, 150, 3, 30, lead_in=6)
        self.assertEqual(len(clips), 2)
        self.assertEqual(clips[0]["start"], 24)
        self.assertIn("WASTED", clips[0]["hook"])
        self.assertTrue(clips[0]["end"] <= clips[1]["start"])


if __name__ == "__main__":
    unittest.main()
