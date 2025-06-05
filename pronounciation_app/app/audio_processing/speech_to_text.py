import whisper

class SpeechRecognizer:
    def __init__(self, model_type="whisper"):
        self.model_type = model_type
        if model_type == "whisper":
            self.model = whisper.load_model("small")
    
    def convert_speech_to_text(self, audio_file, language='english'):
        if self.model_type == "whisper":
            language_code = 'en' if language == 'english' else 'zh'
            result = self.model.transcribe(audio_file, 
                                        language=language_code,
                                        fp16=False)
            return result["text"]