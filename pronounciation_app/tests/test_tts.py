"""
Tests for text-to-speech functionality
"""
import unittest
from app.tts.tts_generator import TTSGenerator
import os

class TestTTS(unittest.TestCase):
    def setUp(self):
        self.tts = TTSGenerator()
    
    def test_tts_initialization(self):
        self.assertIsNotNone(self.tts.engine)
    
    # Note: Skip actual speech tests as they would make noise during testing
    @unittest.skip("Skipping speech test to avoid audio output during testing")
    def test_speak_text(self):
        self.tts.speak_text("Test")

if __name__ == '__main__':
    unittest.main()