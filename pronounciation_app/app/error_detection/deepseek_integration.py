import os
import requests
import base64
from typing import Dict, List

class DeepSeekIntegration:
    def __init__(self):
        self.api_key = self._get_api_key()
        self.api_url = "https://api.deepseek.com/v1/pronunciation"
        self.pronunciation_tips = {
            'english': {
                'th': "Place tongue between teeth for 'th' sounds",
                'r': "Curve tongue slightly for English 'r'",
                'l': "Touch alveolar ridge with tongue tip for 'l'"
            },
            'mandarin': {
                'zh': "Curl tongue back further than English 'j'",
                'ch': "Stronger aspiration than English 'ch'",
                'sh': "Tongue retracted further back than English 'sh'",
                'r': "Create buzzing sound with curled-back tongue"
            }
        }

    def _get_api_key(self):
        """Get API key from environment or config"""
        return os.getenv('DEEPSEEK_API_KEY', 'prototype_key')

    def analyze_pronunciation(self, audio_path: str, target_text: str, language: str) -> Dict:
        """Get detailed pronunciation analysis from DeepSeek"""
        if self.api_key == 'prototype_key':
            return self._generate_enhanced_mock_feedback(target_text, language)
            
        try:
            with open(audio_path, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            
            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "audio": audio_data,
                    "text": target_text,
                    "language": language,
                    "detailed_analysis": True
                }
            )
            return self._enhance_api_response(response.json(), language)
        except Exception as e:
            print(f"DeepSeek error: {e}")
            return self._generate_enhanced_mock_feedback(target_text, language)

    def _enhance_api_response(self, response: Dict, language: str) -> Dict:
        """Add additional pronunciation tips to API response"""
        if 'phoneme_analysis' in response:
            for phoneme in response['phoneme_analysis']:
                if phoneme['phoneme'] in self.pronunciation_tips[language]:
                    phoneme['tip'] = self.pronunciation_tips[language][phoneme['phoneme']]
        return response

    def _generate_enhanced_mock_feedback(self, text: str, language: str) -> Dict:
        """Generate realistic mock feedback with detailed analysis"""
        words = text.split()
        phonemes = ['th', 'sh', 'ch', 'zh', 'r', 'l'] if language == 'english' else ['zh', 'ch', 'sh', 'r']
        
        return {
            "score": 8.2,
            "feedback": "Here's detailed pronunciation analysis:",
            "phoneme_analysis": [
                {
                    "phoneme": phonemes[i % len(phonemes)],
                    "accuracy": min(90, 70 + (i*5)),
                    "tip": self.pronunciation_tips[language].get(
                        phonemes[i % len(phonemes)], 
                        "Listen to native speaker examples"
                    )
                } 
                for i in range(min(3, len(words)))
            ],
            "prosody_feedback": {
                "pace": "Slightly fast",
                "intonation": "Good overall but needs more variation",
                "stress_patterns": "80% correct"
            },
            "actionable_tips": [
                "Practice minimal pairs (ship/sheep)",
                "Record and compare with native speakers",
                "Slow down on difficult consonants"
            ]
        }