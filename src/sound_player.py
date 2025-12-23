"""Sound playback for notifications - simplified and robust implementation"""

import numpy as np
import sounddevice as sd


def _generate_note(freq: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a single bell-like note with harmonics.
    
    Args:
        freq: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate
    
    Returns:
        Audio array
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Generate tone with harmonics for bell-like quality
    tone = (
        np.sin(2 * np.pi * freq * t) +
        0.5 * np.sin(2 * np.pi * freq * 2 * t) +
        0.3 * np.sin(2 * np.pi * freq * 3 * t)
    )
    
    # Apply bell envelope: quick attack, exponential decay
    envelope = np.ones(num_samples)
    attack_samples = int(0.01 * sample_rate)  # 10ms attack
    decay_samples = int(0.15 * sample_rate)     # 150ms decay
    
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    decay_start = attack_samples
    decay_end = min(decay_start + decay_samples, num_samples)
    if decay_end > decay_start:
        decay_curve = np.exp(-np.linspace(0, 4, decay_end - decay_start))
        envelope[decay_start:decay_end] = decay_curve
    
    return tone * envelope


def play_start_sound(duration: float = 0.5, sample_rate: int = 44100):
    """
    Play a pleasant ascending chime sound for start phrase detection.
    
    Args:
        duration: Duration of the sound in seconds
        sample_rate: Sample rate for audio playback
    """
    frequencies = [523.25, 659.25, 783.99]  # C5, E5, G5 (ascending)
    num_notes = len(frequencies)
    note_duration = duration / num_notes
    
    # Calculate exact sample counts to avoid rounding issues
    total_samples = int(sample_rate * duration)
    note_samples = total_samples // num_notes
    
    # Generate each note with exact sample count
    notes = []
    for freq in frequencies:
        note = _generate_note(freq, note_duration, sample_rate)
        # Trim to exact length
        note = note[:note_samples]
        notes.append(note)
    
    # Concatenate notes
    tone = np.concatenate(notes)
    
    # Ensure exact duration (handle any rounding remainder)
    if len(tone) < total_samples:
        tone = np.pad(tone, (0, total_samples - len(tone)), mode='constant')
    elif len(tone) > total_samples:
        tone = tone[:total_samples]
    
    # Normalize and play
    max_val = np.max(np.abs(tone))
    if max_val > 0:
        tone = (tone / max_val) * 0.4  # 40% volume
    
    try:
        sd.play(tone, samplerate=sample_rate, blocking=False)
    except Exception as e:
        # Silently fail - don't interrupt transcription
        pass


def play_stop_sound(duration: float = 0.5, sample_rate: int = 44100):
    """
    Play a pleasant descending chime sound for stop phrase detection.
    
    Args:
        duration: Duration of the sound in seconds
        sample_rate: Sample rate for audio playback
    """
    frequencies = [783.99, 659.25, 523.25]  # G5, E5, C5 (descending)
    num_notes = len(frequencies)
    note_duration = duration / num_notes
    
    # Calculate exact sample counts to avoid rounding issues
    total_samples = int(sample_rate * duration)
    note_samples = total_samples // num_notes
    
    # Generate each note with exact sample count
    notes = []
    for freq in frequencies:
        note = _generate_note(freq, note_duration, sample_rate)
        # Trim to exact length
        note = note[:note_samples]
        notes.append(note)
    
    # Concatenate notes
    tone = np.concatenate(notes)
    
    # Ensure exact duration (handle any rounding remainder)
    if len(tone) < total_samples:
        tone = np.pad(tone, (0, total_samples - len(tone)), mode='constant')
    elif len(tone) > total_samples:
        tone = tone[:total_samples]
    
    # Normalize and play
    max_val = np.max(np.abs(tone))
    if max_val > 0:
        tone = (tone / max_val) * 0.4  # 40% volume
    
    try:
        sd.play(tone, samplerate=sample_rate, blocking=False)
    except Exception as e:
        # Silently fail - don't interrupt transcription
        pass
