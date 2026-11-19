import json
import os
import tempfile
import unittest
from unittest.mock import patch

import find_streams


class ApprovedCreatorTests(unittest.TestCase):
    def test_load_approved_channel_ids_from_json_object(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump({"channel_ids": ["UC123", " UC456 ", ""]}, handle)
            path = handle.name
        try:
            self.assertEqual(find_streams.load_approved_channels(path), {"UC123", "UC456"})
        finally:
            os.unlink(path)

    def test_rejects_invalid_allowlist_shape(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump({"creators": ["UC123"]}, handle)
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                find_streams.load_approved_channels(path)
        finally:
            os.unlink(path)

    def test_find_streams_filters_to_exact_approved_channel_ids(self):
        payload = {"items": [
            {"id": {"videoId": "video1"}, "snippet": {"title": "GTA live", "channelTitle": "Allowed", "channelId": "UC-allowed"}},
            {"id": {"videoId": "video2"}, "snippet": {"title": "GTA live", "channelTitle": "Other", "channelId": "UC-other"}},
        ]}
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
        class ResponseWithJson(Response):
            def __init__(self, data): self.data = data
        with patch("find_streams.urllib.request.urlopen", side_effect=lambda *args, **kwargs: ResponseWithJson(payload)):
            with patch("find_streams.json.load", side_effect=lambda response: response.data):
                streams = find_streams.find_streams("GTA", "test-key", approved_channel_ids={"UC-allowed"})
        self.assertEqual([stream["channel_id"] for stream in streams], ["UC-allowed"])

    def test_rejects_allowlist_without_channel_ids(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump({"creators": ["UC123"]}, handle)
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                find_streams.load_approved_channels(path)
        finally:
            os.unlink(path)

    def test_none_allowlist_keeps_discovery_mode_unfiltered(self):
        payload = {"items": [
            {"id": {"videoId": "video1"}, "snippet": {"title": "GTA live", "channelTitle": "Creator", "channelId": "UC-any"}},
        ]}
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            data = payload
        with patch("find_streams.urllib.request.urlopen", return_value=Response()):
            with patch("find_streams.json.load", side_effect=lambda response: response.data):
                streams = find_streams.find_streams("GTA", "test-key")
        self.assertEqual(len(streams), 1)


if __name__ == "__main__":
    unittest.main()
