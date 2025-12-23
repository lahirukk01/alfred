#!/usr/bin/env python3
"""Entry point script for alfred transcription system"""

import sys
from pathlib import Path

# Add project root to path so we can import src
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main import main

if __name__ == "__main__":
    main()

