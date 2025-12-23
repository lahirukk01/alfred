"""Pattern detection logic for start/end commands"""

from typing import Pattern
from .transcriber_base import TranscriberBase


class PatternDetector:
    """Handles pattern detection for start/end commands"""
    
    def __init__(self, transcriber: TranscriberBase, start_pattern: Pattern, end_pattern: Pattern):
        """
        Initialize the pattern detector.
        
        Args:
            transcriber: Transcriber to use for pattern detection
            start_pattern: Regex pattern for start command
            end_pattern: Regex pattern for end command
        """
        self.transcriber = transcriber
        self.start_pattern = start_pattern
        self.end_pattern = end_pattern
    
    def detect_patterns(self, audio_chunk) -> tuple[bool, bool]:
        """
        Detect start and end patterns in audio chunk.
        
        Args:
            audio_chunk: Audio data to transcribe
        
        Returns:
            Tuple of (start_detected, end_detected)
        """
        segments = self.transcriber.transcribe_audio(audio_chunk)
        if not segments:
            return False, False
        
        full_text = " ".join(segments)
        start_detected = self.start_pattern.search(full_text) is not None
        end_detected = self.end_pattern.search(full_text) is not None
        
        return start_detected, end_detected

