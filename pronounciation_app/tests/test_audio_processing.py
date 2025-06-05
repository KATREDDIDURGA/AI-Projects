"""
Tests for audio processing modules
"""
import unittest
import os
from app.audio_processing.phoneme_converter import PhonemeConverter

class TestPhonemeConverter(unittest.TestCase):
    def setUp(self):
        self.converter = PhonemeConverter()
    
    def test_text_to_phonemes(self):
        # Test with a simple word
        phonemes = self.converter.text_to_phonemes("test")
        self.assertIsInstance(phonemes, list)
        
        # Test with common phoneme combinations
        phonemes = self.converter.text_to_phonemes("that")
        self.assertIn('TH', phonemes)

if __name__ == '__main__':
    unittest.main()