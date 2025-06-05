import os
import json
from typing import List, Dict, Union

def ensure_directories(dirs: List[str]) -> None:
    """Ensure all required directories exist"""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def load_sample_text(language: str) -> List[str]:
    """Load sample texts for pronunciation practice"""
    sample_file = f"data/sample_texts/{language.lower()}.json"
    try:
        with open(sample_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default samples if file not found
        return get_default_samples(language.lower())

def get_default_samples(language: str) -> List[str]:
    """Return default sample texts if file not found"""
    if language == 'english':
        return [
            "The quick brown fox jumps over the lazy dog",
            "She sells seashells by the seashore",
            "How much wood would a woodchuck chuck",
            "Peter Piper picked a peck of pickled peppers",
            "Red lorry, yellow lorry"
        ]
    elif language == 'mandarin':
        return [
            "你好 (nǐ hǎo)",
            "谢谢 (xiè xie)",
            "我爱你 (wǒ ài nǐ)",
            "中国 (zhōng guó)",
            "普通话 (pǔ tōng huà)"
        ]
    return []

def read_api_key(key_name: str) -> str:
    """Read API key from environment or config file"""
    try:
        from app.config import API_KEYS
        return API_KEYS.get(key_name, '')
    except ImportError:
        return os.getenv(key_name, 'prototype_key')

def save_audio_sample(audio_data: bytes, filename: str) -> str:
    """Save audio sample to file"""
    os.makedirs('data/audio_samples', exist_ok=True)
    filepath = os.path.join('data/audio_samples', filename)
    with open(filepath, 'wb') as f:
        f.write(audio_data)
    return filepath