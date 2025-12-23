"""Configuration constants for the whisper transcription system"""

from pathlib import Path

# Get project root (go up from src/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODEL_ROOT = PROJECT_ROOT.parent / "whisper.cpp"

# Model configuration
MODEL_PATH = str(MODEL_ROOT / "models" / "ggml-base.en.bin") if MODEL_ROOT.exists() else ""
MODEL_THREADS = 6
PRINT_PROGRESS = False

# Audio parameters
SAMPLE_RATE = 16000  # Whisper requires 16kHz
CHUNK_DURATION = 3  # Process 3-second chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
OVERLAP_DURATION = 0.5  # Overlap between chunks in seconds (reduced from 1.0)
OVERLAP_SAMPLES = int(SAMPLE_RATE * OVERLAP_DURATION)
BLOCKSIZE_DURATION = 0.5  # Audio capture block size in seconds
BLOCKSIZE = int(SAMPLE_RATE * BLOCKSIZE_DURATION)

# Trigger words/phrases
START_WORD = "hey alexa"
END_WORD = "stop alexa"

