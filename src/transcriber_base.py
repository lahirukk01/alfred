"""Abstract base class for transcribers - allows pluggable transcription backends"""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Union
import numpy as np


class TranscriberBase(ABC):
    """Abstract base class for all transcribers"""
    
    @abstractmethod
    def transcribe_audio(
        self,
        audio_data: Union[np.ndarray, str],
        new_segment_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        Transcribe audio data.
        
        Args:
            audio_data: Audio data as numpy array or file path
            new_segment_callback: Optional callback function called for each new segment
                                  Callback receives (segment_text: str) as argument
        
        Returns:
            List of transcribed text segments (strings)
        """
        pass
    
    @abstractmethod
    def transcribe_stream(
        self,
        audio_stream,
        stop_event,
        buffer: List[str]
    ) -> None:
        """
        Transcribe a continuous audio stream until stop_event is set.
        
        Args:
            audio_stream: Audio stream source (e.g., MicrophoneCapture)
            stop_event: threading.Event that signals when to stop
            buffer: List to append transcribed segments to
        """
        pass

