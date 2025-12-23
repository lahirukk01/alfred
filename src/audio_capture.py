"""Audio capture from microphone"""

import sounddevice as sd
import numpy as np
import queue
import sys
from typing import Callable

from .config import SAMPLE_RATE, BLOCKSIZE


class MicrophoneCapture:
    """Handles microphone audio capture and buffering"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, blocksize: int = BLOCKSIZE):
        """
        Initialize microphone capture.
        
        Args:
            sample_rate: Audio sample rate (default: 16000 Hz)
            blocksize: Size of audio blocks to capture
        """
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.audio_queue = queue.Queue()
        self.stream = None
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function to capture audio from microphone"""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        # Convert to float32 and add to queue
        audio_data = indata[:, 0].astype(np.float32)
        self.audio_queue.put(audio_data.copy())
    
    def start(self):
        """Start the audio input stream"""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.blocksize,
            callback=self._audio_callback
        )
        self.stream.start()
    
    def stop(self):
        """Stop the audio input stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def get_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        """
        Get a chunk of audio from the queue.
        
        Args:
            timeout: Timeout in seconds to wait for audio data
        
        Returns:
            Audio chunk as numpy array, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

