from app.audio_processing.recorder import AudioRecorder
from app.audio_processing.speech_to_text import SpeechRecognizer
from app.error_detection.error_scoring import PronunciationScorer
from app.error_detection.feedback_generator import FeedbackGenerator
from app.tts.tts_generator import TTSGenerator
from app.utils.helpers import ensure_directories, load_sample_text

def main():
    ensure_directories(['data/audio_samples'])
    
    recorder = AudioRecorder()
    recognizer = SpeechRecognizer()
    scorer = PronunciationScorer()
    feedback_gen = FeedbackGenerator()
    tts = TTSGenerator()
    
    print("Welcome to the Pronunciation App with DeepSeek Integration!")
    
    while True:
        print("\nSelect Language:")
        print("1. English")
        print("2. Mandarin")
        language_choice = input("Enter your choice (1-2): ")
        
        if language_choice == '1':
            language = 'english'
        elif language_choice == '2':
            language = 'mandarin'
        else:
            print("Invalid choice. Please try again.")
            continue
        
        print("\nOptions:")
        print("1. Select a sample text")
        print("2. Enter your own text")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '3':
            print("Thank you for using the Pronunciation App!")
            break
        
        target_text = ""
        if choice == '1':
            samples = load_sample_text(language)
            print("\nAvailable samples:")
            for i, sample in enumerate(samples, 1):
                print(f"{i}. {sample[:50]}...")
            
            sample_choice = int(input("Select a sample (number): ")) - 1
            if 0 <= sample_choice < len(samples):
                target_text = samples[sample_choice]
            else:
                print("Invalid selection.")
                continue
        elif choice == '2':
            target_text = input("Enter the text you want to practice: ")
        else:
            print("Invalid choice. Please try again.")
            continue
        
        print(f"\nPlease read the following text:")
        print(f"\n{target_text}\n")
        
        print("Press Enter to start recording...")
        input()
        print("Recording... (Press Enter to stop)")
        audio_file = recorder.record_audio()
        print(f"Recording saved.")
        
        print("Processing speech...")
        recognized_text = recognizer.convert_speech_to_text(audio_file, language)
        print(f"You said: {recognized_text}")
        
        score, errors = scorer.score_pronunciation(target_text, recognized_text, language)
        feedback = feedback_gen.generate_feedback(target_text, recognized_text, errors, language)
        
        print(f"\nYour pronunciation score: {score:.1f}/10.0")
        print("\nFeedback:")
        print(feedback)
        
        speak_feedback = input("\nWould you like to hear the correct pronunciation? (y/n): ")
        if speak_feedback.lower() == 'y':
            tts.speak_text(target_text, language)

if __name__ == "__main__":
    main()