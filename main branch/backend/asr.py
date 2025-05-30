# Enhanced ASR with improved audio preprocessing and error handling
import os
import torch
import logging
import numpy as np
from typing import Optional, List, Dict, Union
import tempfile
import wave
import audioop
import librosa
from scipy import signal
from scipy.io import wavfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Original imports
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Google Cloud Speech-to-Text
try:
    from google.cloud import speech
    GOOGLE_CLOUD_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_SPEECH_AVAILABLE = False
    logger.warning("Google Cloud Speech-to-Text not available.")

# Whisper for multilingual support
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available.")

class ImprovedMultilingualASR:
    def __init__(self, use_premium_asr: bool = True, credentials_path: Optional[str] = None):
        """Initialize Enhanced ASR with better audio processing"""
        self.use_premium = use_premium_asr and GOOGLE_CLOUD_SPEECH_AVAILABLE
        self.cloud_client = None
        
        # Language mapping
        self.language_mapping = {
            'english': 'en-US', 'hindi': 'hi-IN', 'spanish': 'es-ES',
            'french': 'fr-FR', 'german': 'de-DE', 'italian': 'it-IT',
            'portuguese': 'pt-BR', 'russian': 'ru-RU', 'japanese': 'ja-JP',
            'korean': 'ko-KR', 'chinese': 'zh-CN', 'arabic': 'ar-SA',
            'bengali': 'bn-IN', 'tamil': 'ta-IN', 'telugu': 'te-IN',
            'marathi': 'mr-IN', 'gujarati': 'gu-IN', 'kannada': 'kn-IN',
            'malayalam': 'ml-IN', 'punjabi': 'pa-IN', 'urdu': 'ur-IN'
        }
        
        # Initialize ASR components
        if self.use_premium:
            self._initialize_cloud_speech(credentials_path)
        self._initialize_wav2vec2()
        self._initialize_whisper()
    
    def _initialize_cloud_speech(self, credentials_path: Optional[str]):
        """Initialize Google Cloud Speech-to-Text"""
        try:
            if credentials_path and os.path.exists(credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
            self.cloud_client = speech.SpeechClient()
            logger.info("Google Cloud Speech-to-Text initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Speech: {e}")
            self.use_premium = False
            self.cloud_client = None
    
    def _initialize_wav2vec2(self):
        """Initialize Wav2Vec2 with better model"""
        try:
            # Use a better model for improved accuracy
            model_name = "facebook/wav2vec2-large-960h-lv60-self"
            self.asr_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.asr_processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.wav2vec2_available = True
            logger.info("Wav2Vec2 model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Wav2Vec2: {e}")
            self.wav2vec2_available = False
    
    def _initialize_whisper(self):
        """Initialize Whisper model"""
        if not WHISPER_AVAILABLE:
            self.whisper_available = False
            return
        
        try:
            # Use medium model for better accuracy
            model_name = "openai/whisper-medium"
            self.whisper_processor = WhisperProcessor.from_pretrained(model_name)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.whisper_available = True
            logger.info("Whisper model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            self.whisper_available = False
    
    def preprocess_audio(self, audio_data: np.ndarray, target_sr: int = 16000) -> np.ndarray:
        """
        Preprocess audio for better recognition accuracy
        
        Args:
            audio_data: Raw audio data
            target_sr: Target sample rate
            
        Returns:
            Preprocessed audio data
        """
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Apply noise gate to remove background noise
            audio_data = self.apply_noise_gate(audio_data)
            
            # Trim silence from beginning and end
            audio_data = self.trim_silence(audio_data)
            
            # Apply pre-emphasis filter (helps with speech recognition)
            audio_data = self.apply_preemphasis(audio_data)
            
            # Ensure minimum length (some models need minimum audio length)
            min_length = int(0.5 * target_sr)  # 0.5 seconds minimum
            if len(audio_data) < min_length:
                # Pad with zeros if too short
                padding = min_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant')
            
            return audio_data
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}, using original audio")
            return audio_data
    
    def apply_noise_gate(self, audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Apply noise gate to remove low-level background noise"""
        try:
            # Calculate RMS energy in sliding windows
            window_size = 1024
            hop_size = 512
            
            # Pad audio for windowing
            padded_audio = np.pad(audio_data, (window_size//2, window_size//2), mode='reflect')
            
            # Calculate RMS for each window
            rms_values = []
            for i in range(0, len(padded_audio) - window_size + 1, hop_size):
                window = padded_audio[i:i + window_size]
                rms = np.sqrt(np.mean(window**2))
                rms_values.append(rms)
            
            # Interpolate RMS values to match original length
            rms_values = np.array(rms_values)
            if len(rms_values) > 1:
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(rms_values))
                x_new = np.linspace(0, 1, len(audio_data))
                f = interp1d(x_old, rms_values, kind='linear', fill_value='extrapolate')
                rms_interpolated = f(x_new)
            else:
                rms_interpolated = np.full(len(audio_data), rms_values[0] if len(rms_values) > 0 else 0)
            
            # Apply gate
            gate_mask = rms_interpolated > threshold
            gated_audio = audio_data * gate_mask
            
            return gated_audio
            
        except Exception as e:
            logger.warning(f"Noise gate failed: {e}")
            return audio_data
    
    def trim_silence(self, audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from beginning and end"""
        try:
            # Find non-silent regions
            non_silent = np.abs(audio_data) > threshold
            
            if not np.any(non_silent):
                return audio_data
            
            # Find first and last non-silent samples
            first_idx = np.argmax(non_silent)
            last_idx = len(non_silent) - 1 - np.argmax(non_silent[::-1])
            
            # Add small padding
            padding = int(0.1 * 16000)  # 0.1 second padding
            first_idx = max(0, first_idx - padding)
            last_idx = min(len(audio_data) - 1, last_idx + padding)
            
            return audio_data[first_idx:last_idx + 1]
            
        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}")
            return audio_data
    
    def apply_preemphasis(self, audio_data: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter"""
        try:
            return np.append(audio_data[0], audio_data[1:] - alpha * audio_data[:-1])
        except Exception as e:
            logger.warning(f"Pre-emphasis failed: {e}")
            return audio_data
    
    def detect_voice_activity(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Detect if audio contains voice activity
        
        Returns:
            True if voice activity detected, False otherwise
        """
        try:
            # Check if audio has sufficient energy
            rms_energy = np.sqrt(np.mean(audio_data**2))
            if rms_energy < 0.001:  # Very low energy threshold
                return False
            
            # Check for voice-like spectral characteristics
            # Voice typically has energy in 80-3400 Hz range
            if len(audio_data) > 1024:
                fft = np.fft.fft(audio_data[:1024])
                freqs = np.fft.fftfreq(1024, 1/sample_rate)
                
                # Focus on voice frequency range
                voice_range = (freqs >= 80) & (freqs <= 3400)
                voice_energy = np.sum(np.abs(fft[voice_range])**2)
                total_energy = np.sum(np.abs(fft)**2)
                
                voice_ratio = voice_energy / (total_energy + 1e-10)
                
                # If significant energy is in voice range, likely contains speech
                return voice_ratio > 0.1
            
            return True  # Default to True if can't analyze
            
        except Exception as e:
            logger.warning(f"Voice activity detection failed: {e}")
            return True
    
    def speech_to_text_premium(
        self, 
        audio_data: Union[np.ndarray, bytes], 
        language_code: str = "en-US",
        sample_rate: int = 16000,
        **kwargs
    ) -> Dict:
        """Enhanced Google Cloud Speech-to-Text"""
        if not self.cloud_client:
            raise Exception("Google Cloud Speech-to-Text not available")
        
        try:
            # Preprocess audio if it's numpy array
            if isinstance(audio_data, np.ndarray):
                # Check for voice activity first
                if not self.detect_voice_activity(audio_data, sample_rate):
                    return {
                        'transcript': '',
                        'confidence': 0.0,
                        'words': [],
                        'method': 'google_cloud',
                        'warning': 'No voice activity detected'
                    }
                
                # Preprocess audio
                audio_data = self.preprocess_audio(audio_data, sample_rate)
                
                # Convert to bytes
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
            else:
                audio_bytes = audio_data
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            # Enhanced configuration
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code=self.normalize_language_code(language_code),
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                enable_word_confidence=True,
                max_alternatives=3,  # Get multiple alternatives
                model="phone_call",  # Better for varied audio quality
                use_enhanced=True,  # Use enhanced model
                audio_channel_count=1,
                enable_separate_recognition_per_channel=False,
                speech_contexts=[  # Add context for better accuracy
                    speech.SpeechContext(
                        phrases=["hello", "yes", "no", "thank you", "please"],
                        boost=10.0
                    )
                ]
            )
            
            # Perform recognition
            response = self.cloud_client.recognize(config=config, audio=audio)
            
            # Process results
            if response.results:
                result = response.results[0]
                best_alternative = result.alternatives[0]
                
                # Get all alternatives for confidence comparison
                alternatives = []
                for alt in result.alternatives:
                    alternatives.append({
                        'transcript': alt.transcript,
                        'confidence': alt.confidence
                    })
                
                word_info = []
                if best_alternative.words:
                    for word in best_alternative.words:
                        word_info.append({
                            'word': word.word,
                            'confidence': word.confidence,
                            'start_time': word.start_time.total_seconds(),
                            'end_time': word.end_time.total_seconds()
                        })
                
                return {
                    'transcript': best_alternative.transcript,
                    'confidence': best_alternative.confidence,
                    'alternatives': alternatives,
                    'words': word_info,
                    'method': 'google_cloud'
                }
            else:
                return {
                    'transcript': '',
                    'confidence': 0.0,
                    'alternatives': [],
                    'words': [],
                    'method': 'google_cloud',
                    'warning': 'No speech detected'
                }
        
        except Exception as e:
            logger.error(f"Premium ASR failed: {e}")
            raise
    
    def speech_to_text_wav2vec2(self, audio_data: np.ndarray) -> Dict:
        """Enhanced Wav2Vec2 processing"""
        if not self.wav2vec2_available:
            raise Exception("Wav2Vec2 not available")
        
        try:
            # Preprocess audio
            if not self.detect_voice_activity(audio_data):
                return {
                    'transcript': '',
                    'confidence': 0.0,
                    'words': [],
                    'method': 'wav2vec2',
                    'warning': 'No voice activity detected'
                }
            
            processed_audio = self.preprocess_audio(audio_data)
            
            # Process with Wav2Vec2
            inputs = self.asr_processor(
                processed_audio, 
                return_tensors="pt", 
                sampling_rate=16000,
                padding=True
            )
            
            with torch.no_grad():
                logits = self.asr_model(**inputs).logits
            
            # Get predictions with better decoding
            predicted_ids = logits.argmax(dim=-1)
            transcript = self.asr_processor.decode(predicted_ids[0], skip_special_tokens=True)
            
            # Clean up transcript
            transcript = transcript.strip().lower()
            
            return {
                'transcript': transcript,
                'confidence': None,
                'words': [],
                'method': 'wav2vec2'
            }
        
        except Exception as e:
            logger.error(f"Wav2Vec2 ASR failed: {e}")
            raise
    
    def speech_to_text_whisper(self, audio_data: np.ndarray, language: Optional[str] = None) -> Dict:
        """Enhanced Whisper processing"""
        if not self.whisper_available:
            raise Exception("Whisper not available")
        
        try:
            # Preprocess audio
            if not self.detect_voice_activity(audio_data):
                return {
                    'transcript': '',
                    'confidence': 0.0,
                    'words': [],
                    'method': 'whisper',
                    'warning': 'No voice activity detected'
                }
            
            processed_audio = self.preprocess_audio(audio_data)
            
            # Process with Whisper
            input_features = self.whisper_processor(
                processed_audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features
            
            # Generate with better parameters
            generation_kwargs = {
                "max_length": 448,
                "num_beams": 5,  # Use beam search for better accuracy
                "do_sample": False,
                "temperature": 0.0,
                "return_dict_in_generate": True,
                "output_scores": True
            }
            
            if language:
                lang_code = language.split('-')[0]
                forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
                    language=lang_code, task="transcribe"
                )
                generation_kwargs["forced_decoder_ids"] = forced_decoder_ids
            
            # Generate
            generated = self.whisper_model.generate(input_features, **generation_kwargs)
            predicted_ids = generated.sequences
            
            # Decode
            transcript = self.whisper_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            
            return {
                'transcript': transcript,
                'confidence': None,
                'words': [],
                'method': 'whisper'
            }
        
        except Exception as e:
            logger.error(f"Whisper ASR failed: {e}")
            raise
    
    def normalize_language_code(self, lang_code: str) -> str:
        """Normalize language code"""
        if not lang_code:
            return "en-US"
        
        lang_code = lang_code.lower().strip()
        if lang_code in self.language_mapping:
            return self.language_mapping[lang_code]
        
        # Handle simple codes
        simple_to_full = {
            'en': 'en-US', 'hi': 'hi-IN', 'es': 'es-ES', 'fr': 'fr-FR',
            'de': 'de-DE', 'it': 'it-IT', 'pt': 'pt-BR', 'ru': 'ru-RU',
            'ja': 'ja-JP', 'ko': 'ko-KR', 'zh': 'zh-CN', 'ar': 'ar-SA'
        }
        
        if lang_code in simple_to_full:
            return simple_to_full[lang_code]
        
        if '-' in lang_code:
            return lang_code
        
        return "en-US"
    
    def speech_to_text(
        self, 
        audio_data: Union[np.ndarray, bytes], 
        language_code: str = "en-US",
        use_premium: Optional[bool] = None,
        **kwargs
    ) -> str:
        """Main speech-to-text function with improved error handling"""
        should_use_premium = (use_premium if use_premium is not None else self.use_premium)
        
        try:
            # Try premium first
            if should_use_premium and self.cloud_client:
                result = self.speech_to_text_premium(audio_data, language_code, **kwargs)
                if result['transcript'] and result.get('confidence', 0) > 0.5:
                    logger.info(f"Premium ASR success: confidence={result.get('confidence', 'N/A')}")
                    return result['transcript']
                else:
                    logger.warning("Premium ASR returned low confidence or empty result")
            
            # Fallback to local models
            if isinstance(audio_data, np.ndarray):
                # Try Whisper for non-English or as fallback
                if self.whisper_available:
                    try:
                        result = self.speech_to_text_whisper(audio_data, language_code)
                        if result['transcript']:
                            logger.info("Whisper ASR success")
                            return result['transcript']
                    except Exception as e:
                        logger.warning(f"Whisper failed: {e}")
                
                # Try Wav2Vec2 as final fallback
                if self.wav2vec2_available:
                    try:
                        result = self.speech_to_text_wav2vec2(audio_data)
                        if result['transcript']:
                            logger.info("Wav2Vec2 ASR success")
                            return result['transcript']
                    except Exception as e:
                        logger.warning(f"Wav2Vec2 failed: {e}")
            
            # If we get here, all methods failed
            logger.error("All ASR methods failed or returned empty results")
            return ""
        
        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            return ""
    
    def get_detailed_transcription(self, audio_data: Union[np.ndarray, bytes], 
                                 language_code: str = "en-US", **kwargs) -> Dict:
        """Get detailed transcription with all available information"""
        if self.use_premium and self.cloud_client:
            return self.speech_to_text_premium(audio_data, language_code, **kwargs)
        elif isinstance(audio_data, np.ndarray):
            if self.whisper_available:
                return self.speech_to_text_whisper(audio_data, language_code)
            elif self.wav2vec2_available:
                return self.speech_to_text_wav2vec2(audio_data)
        
        raise Exception("No detailed transcription method available")

# Backward compatibility
def speech_to_text(audio_data: np.ndarray) -> str:
    """Original function for backward compatibility"""
    global _asr_instance
    if '_asr_instance' not in globals():
        _asr_instance = ImprovedMultilingualASR()
    
    return _asr_instance.speech_to_text(audio_data)

# Testing and debugging utilities
def test_audio_quality(audio_data: np.ndarray, sample_rate: int = 16000):
    """Test audio quality and provide recommendations"""
    print("=== Audio Quality Analysis ===")
    
    # Basic stats
    print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Data type: {audio_data.dtype}")
    print(f"Min/Max values: {audio_data.min():.4f} / {audio_data.max():.4f}")
    
    # Energy analysis
    rms_energy = np.sqrt(np.mean(audio_data**2))
    print(f"RMS Energy: {rms_energy:.6f}")
    
    if rms_energy < 0.001:
        print("⚠️  WARNING: Very low audio energy - might be too quiet")
    elif rms_energy > 0.5:
        print("⚠️  WARNING: Very high audio energy - might be clipped")
    else:
        print("✅ Audio energy looks good")
    
    # Check for clipping
    clipping_threshold = 0.95
    clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
    clipping_percentage = (clipped_samples / len(audio_data)) * 100
    
    if clipping_percentage > 1:
        print(f"⚠️  WARNING: {clipping_percentage:.1f}% of samples are clipped")
    else:
        print("✅ No significant clipping detected")
    
    # Check duration
    if len(audio_data) / sample_rate < 0.5:
        print("⚠️  WARNING: Audio is very short (< 0.5s) - might affect accuracy")
    elif len(audio_data) / sample_rate > 30:
        print("⚠️  WARNING: Audio is very long (> 30s) - consider splitting")
    else:
        print("✅ Audio duration is appropriate")

if __name__ == "__main__":
    # Example usage
    asr = ImprovedMultilingualASR()
    
    # Test with dummy audio
    dummy_audio = np.random.rand(16000).astype(np.float32) * 0.1  # Quieter random audio
    
    try:
        # Test audio quality first
        test_audio_quality(dummy_audio)
        
        # Test transcription
        transcript = asr.speech_to_text(dummy_audio, "en-US")
        print(f"\nTranscript: '{transcript}'")
        
        # Get detailed results
        detailed = asr.get_detailed_transcription(dummy_audio, "en-US")
        print(f"Detailed result: {detailed}")
        
    except Exception as e:
        print(f"Error: {e}")