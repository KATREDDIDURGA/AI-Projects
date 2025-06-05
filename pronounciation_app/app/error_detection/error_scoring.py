from typing import Dict, List
from jiwer import wer
import eng_to_ipa as ipa
from pypinyin import pinyin, Style
from app.error_detection.deepseek_integration import DeepSeekIntegration

class PronunciationScorer:
    def __init__(self):
        self.deepseek = DeepSeekIntegration()

    def score_pronunciation(self, target_text: str, recognized_text: str, 
                          language: str = 'english') -> tuple[float, List[Dict]]:
        """Calculate pronunciation score with detailed error analysis"""
        # Basic text processing
        target = target_text.lower().strip()
        recognized = recognized_text.lower().strip()
        
        # Get DeepSeek analysis (even in prototype mode)
        deepseek_analysis = self.deepseek.analyze_pronunciation(None, target_text, language)
        
        # Calculate traditional metrics
        error_rate = wer(target, recognized)
        base_score = max(0, 10 * (1 - error_rate))
        
        # Combine with DeepSeek's score (weighted average)
        final_score = (base_score * 0.4) + (deepseek_analysis.get('score', 5) * 0.6)
        
        # Generate detailed errors
        errors = self._generate_phonetic_errors(target_text, recognized_text, language)
        
        return round(final_score, 1), errors

    def _generate_phonetic_errors(self, target_text: str, recognized_text: str,
                                language: str) -> List[Dict]:
        """Identify specific pronunciation errors"""
        errors = []
        target_words = target_text.split()
        recognized_words = recognized_text.split()
        
        for i, (target_word, recognized_word) in enumerate(
            zip(target_words, recognized_words + [''] * (len(target_words) - len(recognized_words)))
        ):
            if not recognized_word:
                errors.append(self._create_missing_word_error(target_word, language))
                continue
            
            if target_word.lower() != recognized_word.lower():
                errors.append(self._create_pronunciation_error(
                    target_word, recognized_word, language
                ))
        
        return errors

    def _create_missing_word_error(self, word: str, language: str) -> Dict:
        """Generate error dict for missing words"""
        return {
            'target': word,
            'recognized': '',
            'target_phonetic': self._get_phonetic(word, language),
            'recognized_phonetic': '',
            'type': 'missing_word'
        }

    def _create_pronunciation_error(self, target: str, recognized: str, 
                                   language: str) -> Dict:
        """Generate error dict for mispronunciations"""
        return {
            'target': target,
            'recognized': recognized,
            'target_phonetic': self._get_phonetic(target, language),
            'recognized_phonetic': self._get_phonetic(recognized, language),
            'type': 'mispronunciation'
        }

    def _get_phonetic(self, word: str, language: str) -> str:
        """Get phonetic representation of a word"""
        if language == 'english':
            return ipa.convert(word)
        elif language == 'mandarin':
            return " ".join([p[0] for p in pinyin(word, style=Style.TONE3)])
        return word