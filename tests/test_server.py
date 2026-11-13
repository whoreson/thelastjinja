"""Unit tests for the daemon's request routing (``--no-stream`` handling)."""

from __future__ import annotations

import unittest

from toolcaller.server import ServerConfig, stream_requested


class TestStreamRequested(unittest.TestCase):
    def test_stream_false_by_default(self):
        self.assertFalse(stream_requested({}, ServerConfig()))

    def test_stream_true_is_honoured_by_default(self):
        self.assertTrue(stream_requested({"stream": True}, ServerConfig()))

    def test_no_stream_overrides_client_stream_true(self):
        config = ServerConfig(no_stream=True)
        self.assertFalse(stream_requested({"stream": True}, config))

    def test_no_stream_keeps_nonstream_nonstream(self):
        config = ServerConfig(no_stream=True)
        self.assertFalse(stream_requested({"stream": False}, config))

    def test_falsy_stream_values(self):
        for value in (None, 0, "", False):
            self.assertFalse(stream_requested({"stream": value}, ServerConfig()))


if __name__ == "__main__":
    unittest.main()
