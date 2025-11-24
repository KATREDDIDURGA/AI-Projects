# Pronunciation Feedback Engine (ASR-Driven Speech ML System)

A modular speech-processing system for evaluating pronunciation, analyzing ASR accuracy, and generating corrective feedback—designed similarly to ASR workflows used in modern language-learning apps.

## 🚀 Features

- 🎤 Real-time speech recording
- 🧠 ASR inference (DeepSeek, Whisper API, Whisper local, Dummy backend)
- 📝 Pronunciation scoring (WER, CER, PER)
- 🗣️ Detailed feedback generation
- 🔊 Offline TTS for reference pronunciation
- 📊 Benchmark utilities for ASR evaluation

## 📁 Project Structure

```
pronunciation-app/
│
├── app/
│   ├── audio_processor.py
│   ├── asr_providers/
│   │   ├── deepseek_asr.py
│   │   ├── whisper_api_asr.py
│   │   ├── whisper_local_asr.py
│   │   └── dummy_asr.py
│   ├── scorer.py
│   ├── feedback_engine.py
│   └── tts_generator.py
│
├── data/
├── models/
├── tests/
│
├── run_app.py
├── requirements.txt
└── README.md
```

## 🛠️ Installation

### Clone the repository

```bash
git clone https://github.com/KATREDDIDURGA/AI-Projects/pronunciation-app.git
cd pronunciation-app
```

### Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### If PyAudio fails

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**Mac:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt install python3-pyaudio
```

## 🔑 Optional: Setup ASR API Keys

Create `.env` in the project root:

```env
DEEPSEEK_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

If no keys are added, the system automatically uses the fallback ASR backend.

## ▶️ Running the App

```bash
python run_app.py
```

**Then:**

1. Choose or type a sentence
2. Record your voice
3. The ASR provider transcribes the speech
4. System scores your pronunciation
5. Generates corrections + native TTS sample

## 📊 Pronunciation Scoring (for ASR evaluation)

The engine computes:

- **WER** – Word Error Rate
- **CER** – Character Error Rate
- **PER** – Phoneme Error Rate
- Substitution / deletion / insertion patterns
- Rate-of-speech analysis

This mirrors the metrics used in production ASR training pipelines.

## 🎧 Feedback Engine

Feedback includes:

- Mispronounced words
- Incorrect phoneme patterns
- Missing / added words
- Stress/timing deviations
- Suggested corrections
- Native audio reference

## 🧠 Model Ready Architecture

The system supports:

✔ Whisper API  
✔ Whisper local models  
✔ DeepSeek backend  
✔ Dummy backend  
✔ Add-your-own ASR provider in 1 file

Designed for fine-tuning, custom datasets, and multilingual experimentation.

## 🚀 Future Enhancements

- Forced alignment (MFA / torchaudio)
- Phoneme-level scoring with wav2vec2
- Whisper fine-tuning pipeline
- Multilingual dataset loading utilities
- Real-time streaming ASR
- Web-based UI

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or feedback, reach out via GitHub issues.

---

**Built with ❤️ for speech ML researchers and language learners**
