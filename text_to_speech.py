import os
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from dotenv import load_dotenv
from pygame import mixer
import pygame

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

def text_to_speech_stream(text, voice_id="pNInz6obpgDQGcFmaJgB"):
    # Perform the text-to-speech conversion
    response = client.text_to_speech.convert(
        voice_id=voice_id, # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Create a BytesIO object to hold the audio data in memory
    audio_stream = BytesIO()

    # Write each chunk of audio data to the stream
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

    # Reset stream position to the beginning
    audio_stream.seek(0)

    # Return the stream for further use
    return audio_stream

if __name__ == "__main__":
    # Initialize pygame mixer
    pygame.init()
    mixer.init()
    
    audio_stream = text_to_speech_stream("Hello, world!", voice_id="pNInz6obpgDQGcFmaJgB")
    # Load and play the audio using pygame
    mixer.music.load(audio_stream)
    mixer.music.play()
    
    # Wait for the audio to finish playing
    while mixer.music.get_busy():
        pygame.time.Clock().tick(10)
