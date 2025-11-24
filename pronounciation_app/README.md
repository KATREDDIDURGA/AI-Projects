Pronunciation Feedback Engine (ASR-Driven Speech ML System)

A modular, experiment-ready speech processing system designed to evaluate pronunciation, generate corrective feedback, and test ASR models for language-learning scenarios.

This project implements a true ASR ML workflow—from audio capture to decoding, scoring, and feedback generation—designed with the same principles used in modern language-learning apps.

🚀 Key Capabilities
🔊 1. Real-Time Speech Capture

Records audio using sounddevice / PyAudio

Automatic trimming, silence detection, RMS normalization

Pluggable preprocessing chain (resample, denoise, etc.)

🧠 2. ASR Inference (Pluggable Backends)

This system supports multiple ASR providers through a unified interface:

DeepSeek Speech API (if key is provided)

OpenAI/Whisper API

Local Whisper models (small/medium/large)

Placeholder backend for offline testing

You can switch providers in one line.

📈 3. Pronunciation Scoring Engine

Implements a multi-metric evaluation pipeline:

WER (Word Error Rate)

CER (Character Error Rate)

PER (Phoneme Error Rate)

Stress & timing heuristics

Syllable alignment

Designed for detailed learner feedback.

🗣️ 4. Feedback Generation

The system generates actionable insights:

Mispronounced phonemes

Missing/added words

Substitutions

Rate-of-speech issues

Segment-level improvement suggestions

🔉 5. Correct Pronunciation Playback

Uses lightweight offline TTS (pyttsx3) to generate:

Native pronunciation reference

Speed-adjusted practice audio

🧪 6. ML Experimentation Ready

Easily plug in Whisper fine-tuned models

Structured dataset directory for future training

Benchmark utilities for evaluating ASR performance

🧩 Project Structure
pronunciation-app/
│── app/
│   ├── audio_processor.py
│   ├── asr_providers/
│   │      ├── deepseek_asr.py
│   │      ├── whisper_api_asr.py
│   │      ├── whisper_local_asr.py
│   │      └── dummy_asr.py
│   ├── scorer.py
│   ├── feedback_engine.py
│   └── tts_generator.py
│
│── data/
│── models/
│── tests/
│── run_app.py
│── requirements.txt
└── README.md

🛠️ Installation
git clone https://github.com/KATREDDIDURGA/AI-Projects/pronunciation-app.git
cd pronunciation-app

Create venv
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt


If PyAudio errors:

Windows → pip install pipwin && pipwin install pyaudio

Mac → brew install portaudio && pip install pyaudio

Linux → sudo apt install python3-pyaudio

Optional ASR Providers

Create .env:

DEEPSEEK_API_KEY=your_key
OPENAI_API_KEY=your_key

▶️ Running
python run_app.py


You will:

Select a phrase

Record audio

Transcription happens via selected ASR provider

Scoring + feedback displayed

Hear native pronunciation
