# Pronunciation App Prototype

A lightweight pronunciation practice application that helps users improve their speaking skills.

## Features

- Record speech and compare to target text
- Get detailed feedback on pronunciation errors
- Hear correct pronunciation examples
- Track your progress over time

## Installation

1. Clone this repository:

git clone https://github.com/yourusername/pronunciation-app.git
cd pronunciation-app
Copy
2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Copy
3. Install dependencies:
pip install -r requirements.txt
Copy
4. Set up API keys (optional):
Create a `.env` file in the root directory with:
DEEPSEEK_API_KEY=your_api_key_here
Copy
## Usage

Run the application:
python run_app.py
Copy
Follow the on-screen instructions to practice pronunciation.

## Project Structure

- `app/`: Main application code
- `data/`: Data storage (audio samples, text samples)
- `models/`: Pre-trained models (for local implementations)
- `tests/`: Unit and integration tests

## Dependencies

- Python 3.7+
- SoundDevice and PyAudio for audio recording
- pyttsx3 for text-to-speech
- Additional dependencies in requirements.txt

## Notes

This is a prototype implementation with simplified functionality. For production use, consider:
- Using larger, more accurate speech recognition models
- Implementing proper phoneme-level error detection
- Adding user accounts and progress tracking
- Developing a graphical user interface
Installation and Setup Guide

Create a new project directory and set up the file structure as shown above
Create a virtual environment:
bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required dependencies:
bashCopypip install numpy sounddevice scipy pyaudio pyttsx3 requests python-dotenv jiwer editdistance

If you face issues with PyAudio installation:

On Windows: pip install pipwin followed by pipwin install pyaudio
On Mac: brew install portaudio then pip install pyaudio
On Linux: sudo apt-get install python3-pyaudio (or equivalent for your distro)


For prototype purposes, no API key is required as the app will use fallback functionality

Running the Application

Make sure you're in the root directory of the project
Run the application:
bashCopypython run_app.py

Follow the on-screen instructions:

Select a sample text or enter your own
Record your pronunciation
Get feedback on your pronunciation
Optionally listen to the correct pronunciation



Notes on Model Selection
The prototype uses:

No speech recognition model: It uses a placeholder API call with a fallback to manual text entry. This avoids loading large models like Whisper or DeepSpeech locally.
DeepSeek integration: The prototype simulates DeepSeek API calls for pronunciation feedback. For a real implementation, you would need to register for an API key.
pyttsx3 for TTS: This is a lightweight offline TTS solution that uses your system's built-in speech engines.

This approach allows you to run the prototype without heavy computational requirements while still demonstrating the core functionality.