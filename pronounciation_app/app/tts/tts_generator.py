from gtts import gTTS
import os

class TTSGenerator:
    def __init__(self):
        pass
    
    def speak_text(self, text, language='english'):
        """
        Speak the given text using gTTS.
        """
        print(f"Speaking: {text}")
        
        lang_code = 'en' if language == 'english' else 'zh-cn'
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save("temp_audio.mp3")
        os.system("afplay temp_audio.mp3")