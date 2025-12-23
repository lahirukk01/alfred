"""Pattern matching utilities for trigger words"""

import re
from typing import Pattern


def create_word_pattern(word: str) -> Pattern:
    """
    Create a case-insensitive regex pattern that ignores commas.
    
    Args:
        word: The word or phrase to create a pattern for (e.g., "hey alexa")
    
    Returns:
        Compiled regex pattern that matches the word case-insensitively,
        allowing commas and whitespace between words.
    
    Examples:
        "hey alexa" matches: "hey alexa", "hey, alexa", "Hey, Alexa", etc.
    """
    # Build pattern manually: replace spaces with pattern that allows optional commas
    # Convert "hey alexa" -> "hey[,\s]+alexa" (matches "hey alexa", "hey, alexa", etc.)
    parts = word.split()
    # Join parts with pattern that matches comma or whitespace
    pattern = r'[,\s]+'.join(re.escape(part) for part in parts)
    # Make it case-insensitive
    return re.compile(pattern, re.IGNORECASE)


def create_patterns(start_word: str, end_word: str) -> tuple[Pattern, Pattern]:
    """
    Create regex patterns for start and end trigger words.
    
    Args:
        start_word: The start trigger phrase
        end_word: The end trigger phrase
    
    Returns:
        Tuple of (start_pattern, end_pattern)
    """
    return create_word_pattern(start_word), create_word_pattern(end_word)

