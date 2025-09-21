# c:\ace3\synthesizer.py

import numpy as np
import torch

class Synthesizer:
    """
    A simple, non-LLM synthesizer that acts as the digital vocal cords.
    It converts output tokens into basic audio waveforms.
    """
    def __init__(self, sample_rate=44100, chunk_duration=0.2):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        
        # Create a time array for generating the waveform
        self.time_array = np.linspace(0., chunk_duration, self.chunk_samples, endpoint=False)
        
        print("🔈 Synthesizer initialized.")

    def generate_waveform(self, tokens: torch.Tensor) -> np.ndarray:
        """
        Generates an audio chunk based on the received tokens.
        This is a simple proof-of-concept generator.
        """
        if tokens is None or tokens.numel() == 0:
            # If there's no impulse, produce silence.
            return np.zeros(self.chunk_samples, dtype=np.float32)

        # Use the first token as the primary driver for the sound
        primary_token = tokens[0].item()

        # --- Organic Sound Generation ---
        # We map the token ID to a frequency within a basic vocal range (e.g., 100Hz to 800Hz)
        # The modulo operator ensures any token number maps to a valid frequency.
        base_frequency = 100 + (primary_token % 700)
        
        # We can use other tokens to add character (e.g., harmonics)
        harmonic_frequency = base_frequency * (2 + (tokens[-1].item() % 3)) if len(tokens) > 1 else 0
        
        # Generate the sine waves for the fundamental and harmonic frequencies
        waveform = 0.3 * np.sin(2 * np.pi * base_frequency * self.time_array)
        if harmonic_frequency > 0:
            waveform += 0.15 * np.sin(2 * np.pi * harmonic_frequency * self.time_array)
            
        # Apply a simple fade-out to prevent clicking
        fade_out = np.linspace(1., 0., self.chunk_samples)
        waveform *= fade_out

        return waveform.astype(np.float32)