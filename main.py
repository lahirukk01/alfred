#!/usr/bin/env python3
"""Entry point script for alfred transcription system"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in project root
project_root = Path(__file__).parent
load_dotenv(project_root / ".env")

# Add project root to path so we can import src
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main import main

if __name__ == "__main__":
    main()

