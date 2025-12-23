"""Command-line interface argument parsing"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Whisper transcription tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Microphone transcription (default)
            python main.py
            
            # File transcription
            python main.py --file audio.mp3
            python main.py --file /path/to/audio.wav
            python main.py --file ../audios/sycopancy_ai_model.mp3
            
            # Verbose mode (show whisper initialization logs)
            python main.py --verbose
            python main.py --file audio.mp3 --verbose
        """
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        metavar="PATH",
        help="Path to audio file to transcribe (relative or absolute)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose whisper initialization logs"
    )
    
    return parser.parse_args()


def resolve_file_path(file_path_str: str) -> Optional[Path]:
    """
    Resolve a file path (relative or absolute) and validate it exists.
    
    Args:
        file_path_str: File path string (relative or absolute)
    
    Returns:
        Resolved Path object if file exists, None otherwise
    
    Raises:
        SystemExit: If file doesn't exist
    """
    file_path = Path(file_path_str)
    
    # Resolve relative paths relative to project root
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    
    # Validate file exists
    if not file_path.exists():
        print(f"Error: Audio file not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    
    return file_path

