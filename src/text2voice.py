"""Text-to-speech using edge-tts"""

import asyncio
import edge_tts
from pathlib import Path
from typing import Optional
from .assets.texts import long_text


async def list_voices():
    """List all available voices"""
    voices = await edge_tts.list_voices()
    for voice in voices:
        if voice["Locale"].startswith("en"):
            print(f"{voice['ShortName']}: {voice['Locale']} - {voice['Gender']}")


async def text_to_speech_async(
    text: str,
    voice: str = "en-US-AriaNeural",
    output_file: Optional[str] = None,
    play: bool = True
) -> bytes:
    """
    Convert text to speech using edge-tts.
    
    Args:
        text: Text to convert to speech
        voice: Voice name (default: en-US-AriaNeural)
        output_file: Optional file path to save audio
        play: Whether to play the audio immediately
    
    Returns:
        Audio data as bytes
    """
    communicate = edge_tts.Communicate(text, voice)
    
    if output_file:
        await communicate.save(output_file)
        print(f"Saved audio to: {output_file}")
    
    # Get audio data
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    
    if play:
        # Play audio using sounddevice
        # Note: edge-tts returns MP3 data, so we'd need to decode it first
        # For simplicity, we can use subprocess with afplay on Mac
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        try:
            subprocess.run(["afplay", tmp_path], check=True)
        finally:
            Path(tmp_path).unlink()
    
    return audio_data


def text_to_speech(
    text: str,
    voice: str = "en-CA-LiamNeural",
    output_file: Optional[str] = None,
    play: bool = True
) -> bytes:
    """
    Synchronous wrapper for text_to_speech_async.
    
    Args:
        text: Text to convert to speech
        voice: Voice name (default: en-US-AriaNeural)
        output_file: Optional file path to save audio
        play: Whether to play the audio immediately
    
    Returns:
        Audio data as bytes
    """
    return asyncio.run(text_to_speech_async(text, voice, output_file, play))


# Example usage
if __name__ == "__main__":
    # List available voices
    # print("Available English voices:")
    # asyncio.run(list_voices())
    # print("\n" + "="*60 + "\n")
    
    # Convert text to speech
    text = long_text
    text_to_speech(text, play=True, voice="en-CA-LiamNeural")
