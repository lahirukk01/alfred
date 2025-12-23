"""Whisper transcription functionality - local whisper.cpp implementation"""

import sys
import os
import threading
from contextlib import contextmanager
from pywhispercpp.model import Model
from pathlib import Path
from typing import Callable, Optional, List, Union
import numpy as np

from .config import MODEL_PATH, MODEL_THREADS, PRINT_PROGRESS, SAMPLE_RATE, CHUNK_SIZE, OVERLAP_SAMPLES
from .transcriber_base import TranscriberBase
from .audio_capture import MicrophoneCapture


@contextmanager
def suppress_output():
    """
    Context manager to suppress both stdout and stderr output.
    Also redirects file descriptors 1 and 2 (stdout and stderr) to /dev/null
    to catch C library output that bypasses Python's sys.stdout/stderr.
    """
    # Save original file descriptors BEFORE any redirection
    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())
    
    # Save Python-level stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Open /dev/null
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    
    try:
        # Redirect file descriptors (catches C library output)
        os.dup2(devnull_fd, 1)  # stdout
        os.dup2(devnull_fd, 2)  # stderr
        
        # Also redirect Python-level streams
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        yield
        
    finally:
        # Flush and close the temporary Python streams first
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.close()
        sys.stderr.close()
        
        # Restore file descriptors
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        
        # Restore Python-level streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Close file descriptors
        os.close(devnull_fd)
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)


class WhisperTranscriber(TranscriberBase):
    """Handles Whisper model loading and transcription - local whisper.cpp implementation"""
    
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        n_threads: int = MODEL_THREADS,
        verbose: bool = False
    ):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_path: Path to the Whisper model file
            n_threads: Number of threads to use for transcription
            verbose: If True, show whisper initialization logs
        """
        self.model_path = model_path
        self.model = None
        self.verbose = verbose
        self._load_model(n_threads)
    
    def _load_model(self, n_threads: int):
        """Load the Whisper model"""
        if self.verbose:
            print(f"Loading model from: {self.model_path}")
            print("-" * 60)
            self.model = Model(self.model_path, n_threads=n_threads, print_progress=PRINT_PROGRESS)
            print(Model.system_info())
            print("-" * 60)
        else:
            # Suppress all output during model loading
            print(f"Loading model from: {self.model_path}")
            print("(Suppressing verbose logs. Use --verbose to see them)")
            print("-" * 60)
            
            with suppress_output():
                self.model = Model(self.model_path, n_threads=n_threads, print_progress=PRINT_PROGRESS)
            
            # Only show system info summary after loading
            print(Model.system_info())
            print("-" * 60)
    
    def transcribe_audio(
        self,
        audio_data: Union[np.ndarray, str],
        new_segment_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        Transcribe audio data.
        
        Args:
            audio_data: Audio data (numpy array or file path)
            new_segment_callback: Optional callback function called for each new segment
                                  Callback receives (segment_text: str) as argument
        
        Returns:
            List of transcribed text segments (strings)
        """
        if new_segment_callback:
            segments = self.model.transcribe(audio_data, new_segment_callback=new_segment_callback)
        else:
            segments = self.model.transcribe(audio_data)
        
        # Convert segment objects to strings
        text_segments = []
        for segment in segments:
            if hasattr(segment, 'text'):
                text = segment.text.strip()
                if text:
                    text_segments.append(text)
            elif isinstance(segment, str):
                text = segment.strip()
                if text:
                    text_segments.append(text)
        
        return text_segments
    
    def transcribe_file(
        self,
        file_path: str,
        new_segment_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to the audio file
            new_segment_callback: Optional callback function called for each new segment
        
        Returns:
            List of transcribed text segments (strings)
        """
        return self.transcribe_audio(file_path, new_segment_callback)
    
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
        audio_buffer = np.array([], dtype=np.float32)
        
        while not stop_event.is_set():
            # Collect audio chunks
            chunk = audio_stream.get_chunk(timeout=0.1)
            if chunk is not None:
                audio_buffer = np.concatenate([audio_buffer, chunk])
            
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

