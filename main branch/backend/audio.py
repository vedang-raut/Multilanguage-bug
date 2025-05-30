# backend/audio.py  For recording and playback.

import sounddevice as sd
import streamlit as st

def record_audio(duration=5, fs=16000):
    st.write(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording finished.")
    return audio_data.flatten()

def display_audio(audio_path):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/mp3")
