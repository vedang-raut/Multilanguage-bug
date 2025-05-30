# Enhanced Text-to-Speech with Google Cloud TTS and gTTS fallback
# Maintains compatibility with existing multilingual project

import os
import logging
from gtts import gTTS
from typing import Optional, Dict, List
import tempfile
import pygame
import time

# Google Cloud TTS (Premium)
try:
    from google.cloud import texttospeech
    GOOGLE_CLOUD_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_TTS_AVAILABLE = False
    logging.warning("Google Cloud TTS not available. Using gTTS only.")

class EnhancedMultilingualTTS:
    def __init__(self, use_premium_tts: bool = True, credentials_path: Optional[str] = None):
        """
        Initialize Enhanced Multilingual TTS
        
        Args:
            use_premium_tts: Whether to use Google Cloud TTS (premium) when available
            credentials_path: Path to Google Cloud service account JSON
        """
        self.use_premium = use_premium_tts and GOOGLE_CLOUD_TTS_AVAILABLE
        self.cloud_client = None
        
        # Language mapping for different TTS services
        self.language_mapping = {
            # Common languages with their codes
            'english': 'en',
            'hindi': 'hi',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'italian': 'it',
            'portuguese': 'pt',
            'russian': 'ru',
            'japanese': 'ja',
            'korean': 'ko',
            'chinese': 'zh',
            'arabic': 'ar',
            'bengali': 'bn',
            'tamil': 'ta',
            'telugu': 'te',
            'marathi': 'mr',
            'gujarati': 'gu',
            'kannada': 'kn',
            'malayalam': 'ml',
            'punjabi': 'pa',
            'urdu': 'ur'
        }
        
        # Google Cloud TTS language codes (more specific)
        self.cloud_language_codes = {
            'en': 'en-US',
            'hi': 'hi-IN',
            'es': 'es-ES',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'it': 'it-IT',
            'pt': 'pt-BR',
            'ru': 'ru-RU',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
            'zh': 'zh-CN',
            'ar': 'ar-XA',
            'bn': 'bn-IN',
            'ta': 'ta-IN',
            'te': 'te-IN',
            'mr': 'mr-IN',
            'gu': 'gu-IN',
            'kn': 'kn-IN',
            'ml': 'ml-IN',
            'pa': 'pa-IN',
            'ur': 'ur-IN'
        }
        
        # Initialize Google Cloud TTS if requested
        if self.use_premium:
            self._initialize_cloud_tts(credentials_path)
        
        # Initialize pygame for audio playback
        try:
            pygame.mixer.init()
            self.audio_available = True
        except:
            self.audio_available = False
            logging.warning("Audio playback not available")
    
    def _initialize_cloud_tts(self, credentials_path: Optional[str]):
        """Initialize Google Cloud TTS client"""
        try:
            if credentials_path and os.path.exists(credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
            self.cloud_client = texttospeech.TextToSpeechClient()
            logging.info("Google Cloud TTS initialized successfully")
        
        except Exception as e:
            logging.error(f"Failed to initialize Google Cloud TTS: {e}")
            self.use_premium = False
            self.cloud_client = None
    
    def normalize_language_code(self, lang_code: str) -> str:
        """
        Normalize language code from various formats
        
        Args:
            lang_code: Language code in various formats
            
        Returns:
            Normalized language code
        """
        # Convert to lowercase and handle common variations
        lang_code = lang_code.lower().strip()
        
        # Handle full language names
        if lang_code in self.language_mapping:
            return self.language_mapping[lang_code]
        
        # Handle language codes with country (e.g., 'en-us' -> 'en')
        if '-' in lang_code:
            return lang_code.split('-')[0]
        
        # Handle common variations
        variations = {
            'eng': 'en',
            'hin': 'hi',
            'spa': 'es',
            'fra': 'fr',
            'deu': 'de',
            'ita': 'it',
            'por': 'pt',
            'rus': 'ru',
            'jpn': 'ja',
            'kor': 'ko',
            'zho': 'zh',
            'ara': 'ar'
        }
        
        return variations.get(lang_code, lang_code)
    
    def get_cloud_language_code(self, lang_code: str) -> str:
        """Get Google Cloud TTS specific language code"""
        normalized = self.normalize_language_code(lang_code)
        return self.cloud_language_codes.get(normalized, f"{normalized}-US")
    
    def text_to_speech_premium(
        self, 
        text: str, 
        lang_code: str, 
        voice_name: Optional[str] = None,
        speaking_rate: float = 1.0,
        pitch: float = 0.0
    ) -> str:
        """
        Generate speech using Google Cloud TTS (Premium)
        
        Args:
            text: Text to convert to speech
            lang_code: Language code
            voice_name: Specific voice name (optional)
            speaking_rate: Speaking rate (0.25 to 4.0)
            pitch: Voice pitch (-20.0 to 20.0)
            
        Returns:
            Path to generated audio file
        """
        if not self.cloud_client:
            raise Exception("Google Cloud TTS not available")
        
        try:
            # Prepare text input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Get appropriate language code
            cloud_lang_code = self.get_cloud_language_code(lang_code)
            
            # Voice selection
            voice_params = {
                "language_code": cloud_lang_code,
                "ssml_gender": texttospeech.SsmlVoiceGender.NEUTRAL
            }
            
            if voice_name:
                voice_params["name"] = voice_name
            
            voice = texttospeech.VoiceSelectionParams(**voice_params)
            
            # Audio configuration
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
                pitch=pitch
            )
            
            # Generate speech
            response = self.cloud_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Save to file
            audio_path = "output.mp3"
            with open(audio_path, "wb") as out:
                out.write(response.audio_content)
            
            logging.info(f"Premium TTS audio saved to {audio_path}")
            return audio_path
            
        except Exception as e:
            logging.error(f"Premium TTS failed: {e}")
            raise
    
    def text_to_speech_free(self, text: str, lang_code: str) -> str:
        """
        Generate speech using gTTS (Free) - Original functionality
        
        Args:
            text: Text to convert to speech
            lang_code: Language code
            
        Returns:
            Path to generated audio file
        """
        try:
            normalized_lang = self.normalize_language_code(lang_code)
            tts = gTTS(text=text, lang=normalized_lang)
            audio_path = "output.mp3"
            tts.save(audio_path)
            logging.info(f"Free TTS audio saved to {audio_path}")
            return audio_path
        
        except Exception as e:
            logging.error(f"Free TTS failed: {e}")
            raise
    
    def text_to_speech(
        self, 
        text: str, 
        lang_code: str, 
        use_premium: Optional[bool] = None,
        **premium_kwargs
    ) -> str:
        """
        Main text-to-speech function (maintains compatibility with existing code)
        
        Args:
            text: Text to convert to speech
            lang_code: Language code
            use_premium: Override default TTS choice
            **premium_kwargs: Additional arguments for premium TTS
            
        Returns:
            Path to generated audio file
        """
        # Determine which TTS to use
        should_use_premium = (use_premium if use_premium is not None else self.use_premium)
        
        try:
            if should_use_premium and self.cloud_client:
                return self.text_to_speech_premium(text, lang_code, **premium_kwargs)
            else:
                return self.text_to_speech_free(text, lang_code)
        
        except Exception as e:
            # Fallback to free TTS if premium fails
            if should_use_premium:
                logging.warning("Premium TTS failed, falling back to free TTS")
                return self.text_to_speech_free(text, lang_code)
            else:
                raise
    
    def play_audio(self, audio_path: str) -> bool:
        """Play generated audio file"""
        if not self.audio_available:
            logging.warning("Audio playback not available")
            return False
        
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            return True
        
        except Exception as e:
            logging.error(f"Audio playback failed: {e}")
            return False
    
    def speak(
        self, 
        text: str, 
        lang_code: str, 
        play_audio: bool = True,
        **kwargs
    ) -> str:
        """
        Convert text to speech and optionally play it
        
        Args:
            text: Text to speak
            lang_code: Language code
            play_audio: Whether to play the audio immediately
            **kwargs: Additional arguments for TTS
            
        Returns:
            Path to generated audio file
        """
        audio_path = self.text_to_speech(text, lang_code, **kwargs)
        
        if play_audio:
            self.play_audio(audio_path)
        
        return audio_path
    
    def get_available_voices(self, lang_code: str) -> List[Dict]:
        """Get available voices for a language (Premium TTS only)"""
        if not self.cloud_client:
            return []
        
        try:
            cloud_lang_code = self.get_cloud_language_code(lang_code)
            voices = self.cloud_client.list_voices(language_code=cloud_lang_code)
            
            voice_list = []
            for voice in voices.voices:
                voice_info = {
                    'name': voice.name,
                    'language': voice.language_codes[0],
                    'gender': voice.ssml_gender.name,
                    'natural_sample_rate': voice.natural_sample_rate_hertz
                }
                voice_list.append(voice_info)
            
            return voice_list
        
        except Exception as e:
            logging.error(f"Error getting voices: {e}")
            return []

# Maintain backward compatibility with existing code
def text_to_speech(text: str, lang_code: str) -> str:
    """
    Original function signature for backward compatibility
    """
    global _tts_instance
    if '_tts_instance' not in globals():
        _tts_instance = EnhancedMultilingualTTS()
    
    return _tts_instance.text_to_speech(text, lang_code)

# Example usage for testing
if __name__ == "__main__":
    # Initialize enhanced TTS
    tts = EnhancedMultilingualTTS()
    
    # Test with different languages
    test_cases = [
        ("Hello, this is a test in English", "en"),
        ("नमस्ते, यह हिंदी में एक परीक्षण है", "hi"),
        ("Hola, esta es una prueba en español", "es"),
        ("Bonjour, ceci est un test en français", "fr")
    ]
    
    for text, lang in test_cases:
        try:
            print(f"Testing {lang}: {text}")
            audio_path = tts.speak(text, lang, play_audio=False)
            print(f"Audio saved to: {audio_path}")
        except Exception as e:
            print(f"Error with {lang}: {e}")