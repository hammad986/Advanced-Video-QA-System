"""
Test to verify the query rewrite module behaves correctly.
Ensures Gemini API is called for short queries and skipped for long queries.
Updated for google-genai SDK (genai.Client instead of genai.GenerativeModel).
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from video_qa.query_rewriter import rewrite_query
from video_qa.config import config

class TestQueryRewriter(unittest.TestCase):

    @patch('video_qa.query_rewriter.genai.Client')
    def test_short_query_rewrites(self, mock_client_class):
        # Mock Gemini client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "What is the main topic of python?"
        mock_response.candidates = [MagicMock()]
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Must temporarily ensure API key is present for test
        import os
        original_key = os.environ.get("GEMINI_API_KEY", "")
        os.environ["GEMINI_API_KEY"] = "test-key"

        result = rewrite_query("python")

        # Restore key
        os.environ["GEMINI_API_KEY"] = original_key

        self.assertEqual(result, "What is the main topic of python?")
        mock_client.models.generate_content.assert_called_once()

    @patch('video_qa.query_rewriter.genai.Client')
    def test_long_query_skipped(self, mock_client_class):
        original = "What did the speaker say about the specific topic of regularization and overfitting?"
        # 13 words — exceeds MAX_WORD_COUNT=8, so rewrite is skipped

        result = rewrite_query(original)

        self.assertEqual(result, original)
        mock_client_class.assert_not_called()

    @patch('video_qa.query_rewriter.genai.Client')
    def test_api_failure_fallback(self, mock_client_class):
        # Mock API failure
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        import os
        original_key = os.environ.get("GEMINI_API_KEY", "")
        os.environ["GEMINI_API_KEY"] = "test-key"

        result = rewrite_query("regularization")

        os.environ["GEMINI_API_KEY"] = original_key

        # Should fallback to original without crashing
        self.assertEqual(result, "regularization")

if __name__ == '__main__':
    unittest.main()
