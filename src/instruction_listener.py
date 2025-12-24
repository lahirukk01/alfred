"""Instruction listening and buffering logic"""

from typing import List, Pattern
from .transcriber_base import TranscriberBase


class InstructionListener:
    """Handles instruction listening and buffering"""
    
    def __init__(self, transcriber: TranscriberBase, start_pattern: Pattern, end_pattern: Pattern):
        """
        Initialize the instruction listener.
        
        Args:
            transcriber: Transcriber to use for instruction transcription
            start_pattern: Regex pattern for start command
            end_pattern: Regex pattern for end command
        """
        self.transcriber = transcriber
        self.start_pattern = start_pattern
        self.end_pattern = end_pattern
        self.instruction_buffer: List[str] = []
        self.is_listening = False
    
    def start_listening(self):
        """Start listening for instructions"""
        self.is_listening = True
        self.instruction_buffer = []
    
    def stop_listening(self):
        """Stop listening for instructions"""
        self.is_listening = False
    
    def process_audio_chunk(self, audio_chunk) -> tuple[List[str], bool]:
        """
        Process an audio chunk and return new segments and whether end pattern was detected.
        
        Args:
            audio_chunk: Audio data to transcribe
        
        Returns:
            Tuple of (new_segments, end_detected)
        """
        if not self.is_listening:
            return [], False
        
        segments = self.transcriber.transcribe_audio(audio_chunk)
        if not segments:
            return [], False
        
        # Combine segments into full text for pattern detection
        new_text = " ".join(segments)
        
        # Check for end pattern
        end_detected = self.end_pattern.search(new_text) is not None
        
        # Buffer all segments directly (overlap removal handled by agent)
        for segment in segments:
            if segment.strip():
                self.instruction_buffer.append(segment)
        
        return segments, end_detected
    
    def get_full_instruction(self) -> str:
        """Get the full buffered instruction"""
        return " ".join(self.instruction_buffer)
    
    def clear_buffer(self):
        """Clear the instruction buffer"""
        self.instruction_buffer = []

