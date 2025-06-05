"""
Audio recording functionality
"""
import os
import time
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import threading

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.audio_dir = "data/audio_samples"
        self.is_recording = False
        self.audio_data = []
        os.makedirs(self.audio_dir, exist_ok=True)
    
    def _record_audio_thread(self):
        """Internal method for recording audio in a separate thread"""
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16') as stream:
            while self.is_recording:
                chunk, _ = stream.read(self.sample_rate)
                self.audio_data.append(chunk)

    def _process_audio(self, audio_data):
        """Normalize and clean audio before saving"""
        audio_data = audio_data.astype(np.float32)
        audio_data /= np.max(np.abs(audio_data))  # Normalize
        return audio_data
    
    def record_audio(self, duration=None):
        """
        Record audio from the microphone
        If duration is None, will record until Enter is pressed
        Returns the path to the saved audio file
        """
        filename = f"{self.audio_dir}/recording_{int(time.time())}.wav"
        
        if duration:
            # Fixed duration recording
            print(f"Recording for {duration} seconds...")
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16'
            )
            sd.wait()
        else:
            # Record until Enter is pressed
            print("Recording... Press Enter to stop.")
            
            # Initialize recording state
            self.is_recording = True
            self.audio_data = []
            
            # Start recording in a separate thread
            recording_thread = threading.Thread(target=self._record_audio_thread)
            recording_thread.start()
            
            # Wait for Enter key
            input()
            
            # Stop recording
            self.is_recording = False
            recording_thread.join()
            
            # Convert list of chunks to numpy array
            audio_data = np.concatenate(self.audio_data, axis=0)
        
        # Save the recording
        wavfile.write(filename, self.sample_rate, audio_data)
        print(f"Recording saved to {filename}")
        return filename