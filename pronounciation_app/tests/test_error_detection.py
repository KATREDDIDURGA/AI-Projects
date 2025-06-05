"""
Tests for error detection modules
"""
import unittest
from app.error_detection.error_scoring import PronunciationScorer
from app.error_detection.feedback_generator import FeedbackGenerator

class TestErrorDetection(unittest.TestCase):
    def setUp(self):
        self.scorer = PronunciationScorer()
        self.feedback_gen = FeedbackGenerator()
    
    def test_perfect_match(self):
        target = "this is a test"
        recognized = "this is a test"
        score, errors = self.scorer.score_pronunciation(target, recognized)
        self.assertAlmostEqual(score, 10.0)
        self.assertEqual(len(errors), 0)
    
    def test_partial_match(self):
        target = "this is a test"
        recognized = "this is test"
        score, errors = self.scorer.score_pronunciation(target, recognized)
        self.assertLess(score, 10.0)
        self.assertGreater(len(errors), 0)
    
    def test_feedback_generation(self):
        target = "the quick brown fox"
        recognized = "the quick brown"
        _, errors = self.scorer.score_pronunciation(target, recognized)
        feedback = self.feedback_gen.generate_feedback(target, recognized, errors)
        self.assertIsInstance(feedback, str)
        self.assertGreater(len(feedback), 0)

if __name__ == '__main__':
    unittest.main()