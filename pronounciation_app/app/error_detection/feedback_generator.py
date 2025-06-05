from typing import Dict, List
from app.error_detection.deepseek_integration import DeepSeekIntegration

class FeedbackGenerator:
    def __init__(self):
        self.deepseek = DeepSeekIntegration()
        self.tone_descriptions = {
            '1': 'Flat (high, steady)',
            '2': 'Rising (like asking "What?")',
            '3': 'Dipping (fall then rise)',
            '4': 'Falling (sharp drop)',
            '5': 'Neutral (light and quick)'
        }

    def generate_feedback(self, target_text: str, recognized_text: str, 
                        errors: List[Dict], language: str) -> str:
        """Generate comprehensive feedback using DeepSeek analysis"""
        if not recognized_text.strip():
            return "🔇 No speech detected. Please try again."
            
        analysis = self.deepseek.analyze_pronunciation(None, target_text, language)
        
        feedback = [
            f"# 🎯 Pronunciation Feedback",
            f"## 📊 Overall Score: {analysis.get('score', 0):.1f}/10.0",
            "",
            "## 🎙️ What You Said:",
            f"_{recognized_text}_",
            "",
            "## 🎯 Target Text:",
            f"_{target_text}_",
            ""
        ]

        if language == 'mandarin':
            feedback.extend(self._generate_mandarin_section(target_text, analysis))
        else:
            feedback.extend(self._generate_english_section(analysis))

        if errors:
            feedback.extend([
                "",
                "## 🔍 Detailed Error Analysis:",
                *self._format_errors(errors)
            ])

        feedback.extend([
            "",
            "## 💡 Actionable Tips:",
            *[f"- {tip}" for tip in analysis.get('actionable_tips', [])],
            "",
            "🔊 Try recording again while focusing on one improvement at a time."
        ])

        return "\n".join(feedback)

    def _generate_mandarin_section(self, text: str, analysis: Dict) -> List[str]:
        """Generate Mandarin-specific feedback"""
        section = [
            "## 🇨🇳 Mandarin Focus Areas:",
            ""
        ]
        
        if 'phoneme_analysis' in analysis:
            section.append("### 🗣️ Tone and Consonant Accuracy:")
            for item in analysis['phoneme_analysis']:
                section.append(
                    f"- **{item['phoneme'].upper()}**: {item['accuracy']}% correct "
                    f"(Tip: {item.get('tip', 'Practice with native recordings')})"
                )
        
        chinese_part = text.split(' (')[0] if ' (' in text else text
        pinyin_part = text.split(' (')[1].split(')')[0] if ' (' in text else self._get_pinyin(chinese_part)
        
        section.extend([
            "",
            "### 🎚 Tone Breakdown:",
            *[f"- {char} → {pinyin} ({self.tone_descriptions.get(pinyin[-1], 'Neutral')})" 
              for char, pinyin in zip(chinese_part, pinyin_part.split())]
        ])
        
        return section

    def _generate_english_section(self, analysis: Dict) -> List[str]:
        """Generate English-specific feedback"""
        section = [
            "## 🇬🇧 English Focus Areas:",
            ""
        ]
        
        if 'phoneme_analysis' in analysis:
            section.append("### 🗣️ Problem Sounds:")
            for item in analysis['phoneme_analysis']:
                section.append(
                    f"- **{item['phoneme'].upper()}**: {item['accuracy']}% correct "
                    f"(Tip: {item.get('tip', '')})"
                )
        
        if 'prosody_feedback' in analysis:
            section.extend([
                "",
                "### 🎭 Speech Rhythm:",
                f"- Pace: {analysis['prosody_feedback'].get('pace', 'Good')}",
                f"- Intonation: {analysis['prosody_feedback'].get('intonation', 'Good')}",
                f"- Word Stress: {analysis['prosody_feedback'].get('stress_patterns', 'Good')}"
            ])
        
        return section

    def _format_errors(self, errors: List[Dict]) -> List[str]:
        """Format error analysis"""
        formatted = []
        for error in errors:
            if error['recognized']:
                formatted.append(
                    f"- ❌ Said '{error['recognized']}' instead of '{error['target']}' "
                    f"(Phonetic: {error['recognized_phonetic']} vs {error['target_phonetic']})"
                )
            else:
                formatted.append(f"- ❗ Missing word: '{error['target']}'")
        return formatted

    def _get_pinyin(self, text: str) -> str:
        """Convert Chinese characters to pinyin"""
        from pypinyin import pinyin, Style
        return " ".join([p[0] for p in pinyin(text, style=Style.TONE3)])