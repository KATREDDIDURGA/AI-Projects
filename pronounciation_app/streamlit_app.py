import streamlit as st
import time
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import os
from app.utils.helpers import load_sample_text
from app.audio_processing.speech_to_text import SpeechRecognizer
from app.error_detection.error_scoring import PronunciationScorer
from app.error_detection.feedback_generator import FeedbackGenerator
from app.tts.tts_generator import TTSGenerator
from app.error_detection.deepseek_integration import DeepSeekIntegration

# Initialize components
recognizer = SpeechRecognizer()
scorer = PronunciationScorer()
feedback_gen = FeedbackGenerator()
tts = TTSGenerator()
deepseek = DeepSeekIntegration()

# Constants
SAMPLE_RATE = 16000

# Initialize session state
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = []
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

def record_audio():
    st.session_state.audio_data = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while st.session_state.recording:
            chunk, _ = stream.read(SAMPLE_RATE)
            st.session_state.audio_data.append(chunk)
            time.sleep(0.1)

def stop_recording():
    st.session_state.recording = False
    if st.session_state.audio_data:
        audio_data = np.concatenate(st.session_state.audio_data, axis=0)
        filename = f"data/audio_samples/recording_{int(time.time())}.wav"
        wavfile.write(filename, SAMPLE_RATE, audio_data)
        st.session_state.audio_file = filename
        st.success(f"Recording saved to {filename}")
    else:
        st.error("No audio data recorded.")

def run_streamlit_app():
    st.title("Pronunciation Practice App with DeepSeek")
    
    language = st.radio("Select Language", ["English", "Mandarin"])
    text_option = st.radio("Choose Text Option", ["Select Sample Text", "Enter Your Own Text"])
    
    if text_option == "Select Sample Text":
        samples = load_sample_text(language.lower())
        sample_choice = st.selectbox("Select a sample text", samples)
        target_text = sample_choice
    else:
        target_text = st.text_input("Enter the text you want to practice:")
    
    if not st.session_state.recording:
        if st.button("Start Recording"):
            st.session_state.recording = True
            st.session_state.audio_file = None
            st.rerun()
    else:
        if st.button("Stop Recording"):
            stop_recording()
            st.rerun()
    
    if st.session_state.recording:
        st.write("Recording... Click 'Stop Recording' to finish.")
        record_audio()
    
    if st.session_state.audio_file:
        st.write(f"Processing recording: {st.session_state.audio_file}")
        
        recognized_text = recognizer.convert_speech_to_text(st.session_state.audio_file, language.lower())
        st.write(f"You said: {recognized_text}")
        
        score, errors = scorer.score_pronunciation(target_text, recognized_text, language.lower())
        feedback = feedback_gen.generate_feedback(target_text, recognized_text, errors, language.lower())
        
        st.write(f"Your pronunciation score: {score:.1f}/10.0")
        st.write("Feedback:")
        st.write(feedback)
        
        if st.button("Hear Correct Pronunciation"):
            tts.speak_text(target_text, language.lower())

if __name__ == "__main__":
    run_streamlit_app()