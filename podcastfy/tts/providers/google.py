"""Google Cloud Text-to-Speech provider implementation."""

from google.cloud import texttospeech
from typing import List
from ..base import TTSProvider
import logging

logger = logging.getLogger(__name__)

class GoogleTTS(TTSProvider):
    """Google Cloud Text-to-Speech provider."""

    def __init__(self, api_key: str = None, model: str = "en-US-Wavenet-D"):
        """
        Initialize Google Cloud TTS provider.

        Args:
            api_key (str): Google Cloud API key.
            model (str): Default voice model to use (e.g., 'en-US-Wavenet-D').
        """
        self.model = model
        try:
            self.client = texttospeech.TextToSpeechClient(
                client_options={'api_key': api_key} if api_key else None
            )
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS client: {str(e)}")
            raise

    def generate_audio(self, text: str, voice: str = "en-US-Wavenet-D",
                      model: str = None, **kwargs) -> bytes:
        """
        Generate audio using Google Cloud TTS API.

        Args:
            text (str): Text to convert to speech.
            voice (str): Voice name to use (e.g., 'en-US-Wavenet-D'). This will be used as the 'name'.
            model (str): This parameter is kept for consistency but the 'voice' parameter determines the model.

        Returns:
            bytes: Audio data.
        """
        self.validate_parameters(text, voice, model or self.model)

        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # The voice name itself specifies the model (e.g., Wavenet, standard)
            language_code = "-".join(voice.split("-")[:2])

            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            )

            return response.audio_content

        except Exception as e:
            logger.error(f"Failed to generate audio with Google TTS: {str(e)}")
            raise RuntimeError(f"Failed to generate audio with Google TTS: {str(e)}") from e

    def get_supported_tags(self) -> List[str]:
        """Get supported SSML tags."""
        return self.COMMON_SSML_TAGS
    def validate_parameters(self, text: str, voice: str, model: str) -> None:
        """
        Validate input parameters before generating audio.
        
        Args:
            text (str): Input text
            voice (str): Voice ID/name
            model (str): Model name
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().validate_parameters(text, voice, model)
        
        if not text:
            raise ValueError("Text cannot be empty")
        
        if not voice:
            raise ValueError("Voice must be specified")                                                                             