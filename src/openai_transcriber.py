"""OpenAI Whisper API transcriber implementation"""

import os
import sys
import numpy as np
import threading
from typing import List, Optional, Callable, Union
from openai import OpenAI

from .transcriber_base import TranscriberBase
from .config import SAMPLE_RATE, CHUNK_DURATION, CHUNK_SIZE, OVERLAP_SAMPLES
from .audio_capture import MicrophoneCapture


class OpenAITranscriber(TranscriberBase):
    """Transcriber using OpenAI Whisper API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        """
        Initialize OpenAI transcriber.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Whisper model to use (default: whisper-1)
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def transcribe_audio(
        self,
        audio_data: Union[np.ndarray, str],
        new_segment_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        Transcribe audio data using OpenAI Whisper API.
        
        Args:
            audio_data: Audio data as numpy array or file path
            new_segment_callback: Optional callback function called for each new segment
        
        Returns:
            List of transcribed text segments
        """
        # Handle numpy array - save to temporary file
        if isinstance(audio_data, np.ndarray):
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                # Convert to int16 for WAV format
                audio_int16 = (audio_data * 32767).astype(np.int16)
                sf.write(tmp_path, audio_int16, SAMPLE_RATE)
            
            try:
                with open(tmp_path, "rb") as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        response_format="verbose_json"
                    )
                
                # Extract segments
                segments = []
                if hasattr(transcript, 'segments') and transcript.segments:
                    for seg in transcript.segments:
                        text = seg.get('text', '').strip()
                        if text:
                            segments.append(text)
                            if new_segment_callback:
                                new_segment_callback(text)
                else:
                    # Fallback: use full text if no segments
                    full_text = getattr(transcript, 'text', '').strip()
                    if full_text:
                        segments.append(full_text)
                        if new_segment_callback:
                            new_segment_callback(full_text)
                
                return segments
            finally:
                os.unlink(tmp_path)
        
        else:
            # File path
            with open(audio_data, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="verbose_json"
                )
            
            segments = []
            if hasattr(transcript, 'segments') and transcript.segments:
                for seg in transcript.segments:
                    text = seg.get('text', '').strip()
                    if text:
                        segments.append(text)
                        if new_segment_callback:
                            new_segment_callback(text)
            else:
                full_text = getattr(transcript, 'text', '').strip()
                if full_text:
                    segments.append(full_text)
                    if new_segment_callback:
                        new_segment_callback(full_text)
            
            return segments
    
    def transcribe_stream(
        self,
        audio_stream: MicrophoneCapture,
        stop_event: threading.Event,
        buffer: List[str]
    ) -> None:
        """
        Transcribe a continuous audio stream until stop_event is set.
        
        Args:
            audio_stream: MicrophoneCapture instance
            stop_event: threading.Event that signals when to stop
            buffer: List to append transcribed segments to
        """
        import tempfile
        import soundfile as sf
        
        audio_buffer = np.array([], dtype=np.float32)
        chunk_buffer = []
        
        while not stop_event.is_set():
            # Collect audio chunks
            chunk = audio_stream.get_chunk(timeout=0.1)
            if chunk is not None:
                audio_buffer = np.concatenate([audio_buffer, chunk])
                chunk_buffer.append(chunk)
            
            # Process when we have enough audio (or when stopping)
            if len(audio_buffer) >= CHUNK_SIZE or (stop_event.is_set() and len(audio_buffer) > 0):
                if len(audio_buffer) > 0:
                    # Extract chunk for transcription
                    chunk_to_transcribe = audio_buffer[:CHUNK_SIZE] if len(audio_buffer) >= CHUNK_SIZE else audio_buffer
                    
                    # Keep overlap
                    if len(audio_buffer) >= CHUNK_SIZE:
                        audio_buffer = audio_buffer[CHUNK_SIZE - OVERLAP_SAMPLES:]
                    else:
                        audio_buffer = np.array([], dtype=np.float32)
                    
                    # Transcribe the chunk
                    try:
                        segments = self.transcribe_audio(chunk_to_transcribe)
                        for segment in segments:
                            if segment.strip():
                                buffer.append(segment)
                    except Exception as e:
                        print(f"Error during transcription: {e}", file=sys.stderr)
        
        # Process any remaining audio
        if len(audio_buffer) > 0 and not stop_event.is_set():
            try:
                segments = self.transcribe_audio(audio_buffer)
                for segment in segments:
                    if segment.strip():
                        buffer.append(segment)
            except Exception as e:
                print(f"Error during final transcription: {e}", file=sys.stderr)

