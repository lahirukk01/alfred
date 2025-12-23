"""Instruction listening and buffering logic"""

from typing import List, Pattern
from .transcriber_base import TranscriberBase


class InstructionListener:
    """Handles instruction listening, buffering, and overlap removal"""
    
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
        self.last_transcribed_text = ""
        self.is_listening = False
    
    def start_listening(self):
        """Start listening for instructions"""
        self.is_listening = True
        self.instruction_buffer = []
        self.last_transcribed_text = ""
    
    def stop_listening(self):
        """Stop listening for instructions"""
        self.is_listening = False
        self.last_transcribed_text = ""
    
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
        
        # Combine segments into full text
        new_text = " ".join(segments)
        
        # Remove overlapping text
        new_segments = self._remove_overlap(new_text)
        
        # Check for end pattern
        end_detected = self.end_pattern.search(new_text) is not None
        
        # Update last transcribed text for overlap detection
        self.last_transcribed_text = new_text
        
        # Buffer new segments
        for segment in new_segments:
            if segment.strip():
                self.instruction_buffer.append(segment)
        
        return new_segments, end_detected
    
    def _remove_overlap(self, new_text: str) -> List[str]:
        """
        Remove overlapping text from the beginning of new_text.
        
        Args:
            new_text: New transcribed text
        
        Returns:
            List of non-overlapping segments
        """
        if not self.last_transcribed_text:
            # First chunk, no overlap to remove
            return [new_text] if new_text.strip() else []
        
        # Split into words for overlap detection
        last_words = self.last_transcribed_text.split()
        new_words = new_text.split()
        
        # Find the longest overlap (matching words from the end)
        overlap_len = 0
        for i in range(1, min(len(last_words), len(new_words)) + 1):
            if last_words[-i:] == new_words[:i]:
                overlap_len = i
        
        # Remove overlapping words from new text
        if overlap_len > 0:
            new_text_clean = " ".join(new_words[overlap_len:])
        else:
            new_text_clean = new_text
        
        return [new_text_clean] if new_text_clean.strip() else []
    
    def get_full_instruction(self) -> str:
        """Get the full buffered instruction"""
        return " ".join(self.instruction_buffer)
    
    def clear_buffer(self):
        """Clear the instruction buffer"""
        self.instruction_buffer = []

