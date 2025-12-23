"""Main entry point for whisper transcription"""

import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

from .config import (
    CHUNK_DURATION,
    CHUNK_SIZE,
    OVERLAP_SAMPLES,
    START_WORD,
    END_WORD,
)
from .audio_capture import MicrophoneCapture
from .transcriber import WhisperTranscriber
from .openai_transcriber import OpenAITranscriber
from .pattern_matcher import create_patterns
from .cli import parse_args, resolve_file_path
from .sound_player import play_start_sound, play_stop_sound
from .pattern_detector import PatternDetector
from .instruction_listener import InstructionListener


def filter_command_texts(text: str, start_pattern, end_pattern) -> str:
    """
    Remove start/end command texts from the instruction while preserving formatting.
    
    Args:
        text: The full instruction text
        start_pattern: Compiled regex pattern for start command
        end_pattern: Compiled regex pattern for end command
    
    Returns:
        Text with command phrases removed, preserving case and symbols
    """
    import re
    # Remove start pattern matches (case-insensitive but preserve original case)
    text = start_pattern.sub('', text)
    # Remove end pattern matches
    text = end_pattern.sub('', text)
    # Clean up extra whitespace but preserve structure
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = text.strip()
    return text


def transcribe_microphone(
    pattern_detector: PatternDetector,
    instruction_listener: InstructionListener,
    start_pattern,
    end_pattern
):
    """
    Continuously transcribe audio from microphone with start/end pattern detection.
    
    When start pattern is detected:
    - Start buffering instructions
    
    When end pattern is detected:
    - Print the buffered instruction transcription (with command texts removed)
    """
    print("Starting microphone transcription...")
    print(f"Listening window: {CHUNK_DURATION} seconds")
    print(f"Start phrase: '{START_WORD}' | End phrase: '{END_WORD}'")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    print("-" * 60)
    
    with MicrophoneCapture() as mic:
        print("Listening for trigger phrases...")
        
        audio_buffer = np.array([], dtype=np.float32)
        
        try:
            while True:
                # Collect audio chunks until we have enough for transcription
                while len(audio_buffer) < CHUNK_SIZE:
                    chunk = mic.get_chunk(timeout=0.1)
                    if chunk is not None:
                        audio_buffer = np.concatenate([audio_buffer, chunk])
                    else:
                        continue
                
                # Extract chunk for transcription
                chunk_to_transcribe = audio_buffer[:CHUNK_SIZE]
                # Keep some overlap for better continuity
                audio_buffer = audio_buffer[CHUNK_SIZE - OVERLAP_SAMPLES:]
                
                try:
                    if instruction_listener.is_listening:
                        # Instruction listening mode
                        new_segments, end_detected = instruction_listener.process_audio_chunk(chunk_to_transcribe)
                        
                        # Print new segments in real-time
                        for segment in new_segments:
                            if segment.strip():
                                print(segment, end=" ", flush=True)
                        
                        # Handle end pattern detection
                        if end_detected:
                            _handle_end_pattern(
                                instruction_listener,
                                start_pattern,
                                end_pattern
                            )
                    else:
                        # Pattern detection mode
                        start_detected, end_detected = pattern_detector.detect_patterns(chunk_to_transcribe)
                        
                        if start_detected:
                            _handle_start_pattern(instruction_listener)
                        # Note: end_detected in pattern mode shouldn't happen, but handle it just in case
                        if end_detected and instruction_listener.is_listening:
                            _handle_end_pattern(
                                instruction_listener,
                                start_pattern,
                                end_pattern
                            )
                            
                except Exception as e:
                    print(f"\nError during transcription: {e}", file=sys.stderr)
                    
        except KeyboardInterrupt:
            print("\n\nStopping transcription...")
            print("=" * 60)


def _handle_start_pattern(instruction_listener: InstructionListener):
    """Handle start pattern detection"""
    print(f"\n✓ START pattern detected: '{START_WORD}'")
    play_start_sound()
    instruction_listener.start_listening()
    print("→ Listening for instructions...")


def _handle_end_pattern(
    instruction_listener: InstructionListener,
    start_pattern,
    end_pattern
):
    """Handle end pattern detection"""
    print(f"\n✓ END pattern detected: '{END_WORD}'")
    play_stop_sound()
    
    # Get and clean the instruction
    if instruction_listener.instruction_buffer:
        full_instruction = instruction_listener.get_full_instruction()
        cleaned_instruction = filter_command_texts(
            full_instruction,
            start_pattern,
            end_pattern
        )
        print("\n" + "=" * 60)
        print("INSTRUCTION TRANSCRIPTION:")
        print("-" * 60)
        print(cleaned_instruction)
        print("=" * 60 + "\n")
    else:
        print("\n(No instruction detected)\n")
    
    # Reset for next cycle
    instruction_listener.stop_listening()
    instruction_listener.clear_buffer()
    print("Listening for trigger phrases...")


def transcribe_file(transcriber, file_path: str):
    """Transcribe an audio file"""
    print(f"Transcribing file: {Path(file_path).name}")
    print("Partial results (as they come in):")
    print("-" * 60)
    
    def on_new_segment(segment):
        if isinstance(segment, str):
            print(segment, end="", flush=True)
        else:
            print(segment, end="", flush=True)
    
    segments = transcriber.transcribe_file(file_path, new_segment_callback=on_new_segment)
    
    print("\n" + "=" * 60)
    print("TRANSCRIPTION COMPLETE")
    print(f"Total segments: {len(segments)}")
    print("=" * 60)


def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize local transcriber for pattern detection (cost-effective)
    pattern_transcriber = WhisperTranscriber(verbose=args.verbose)
    print("Using local whisper model for pattern detection")
    
    # Initialize OpenAI transcriber for instruction transcription (high quality)
    # Only used during instruction listening to save costs
    instruction_transcriber = None
    try:
        instruction_transcriber = OpenAITranscriber()
        print("Using OpenAI Whisper API for instruction transcription")
    except ValueError as e:
        print(f"Warning: {e}")
        print("Falling back to local model for instructions too")
        instruction_transcriber = pattern_transcriber
    
    # Create trigger word patterns
    start_pattern, end_pattern = create_patterns(START_WORD, END_WORD)
    
    # Initialize pattern detector (uses local model) and instruction listener (uses OpenAI)
    pattern_detector = PatternDetector(pattern_transcriber, start_pattern, end_pattern)
    instruction_listener = InstructionListener(instruction_transcriber, start_pattern, end_pattern)
    
    # Check if user wants to transcribe file or microphone
    if args.file:
        # Resolve and validate file path
        file_path = resolve_file_path(args.file)
        # File transcription mode (use local model for file transcription)
        transcribe_file(pattern_transcriber, str(file_path))
    else:
        # Microphone transcription mode (default)
        transcribe_microphone(
            pattern_detector,
            instruction_listener,
            start_pattern,
            end_pattern
        )


if __name__ == "__main__":
    main()
