# vocalsynth.py

import os
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Single consolidated state file for all modules
GLOBAL_STATE_FILE = Path(__file__).resolve().parent.parent / "state" / "state.pkl"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# Biquad Filter (Helper for Formant Synthesis)
# A direct form 2 transposed biquad filter implementation for high precision.
# ==============================================================================

class BiquadFilter:
    """
    Implements a biquad (second-order) IIR filter. Used for modeling formants.
    Coefficients: b0, b1, b2 (numerator) and a0, a1, a2 (denominator).
    """
    def __init__(self, b_coeffs: np.ndarray, a_coeffs: np.ndarray):
        # Ensure coefficients are float64
        self.b = b_coeffs.astype(np.float64) # [b0, b1, b2]
        self.a = a_coeffs.astype(np.float64) # [a0, a1, a2] (a0 is usually 1.0)
        
        # Internal state variables (for direct form 2 transposed)
        self.z1 = np.float64(0.0)
        self.z2 = np.float64(0.0)

        # Normalize by a0 if it's not 1.0 (though for resonant filters, it usually is)
        if self.a[0] != 1.0:
            self.b /= self.a[0]
            self.a /= self.a[0]

    def process_sample(self, x: np.float64) -> np.float64:
        """Processes a single audio sample."""
        # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        # Direct form 2 transposed
        y = self.b[0] * x + self.z1
        self.z1 = self.b[1] * x - self.a[1] * y + self.z2
        self.z2 = self.b[2] * x - self.a[2] * y
        return y

    def reset_state(self):
        """Resets the internal filter state."""
        self.z1 = np.float64(0.0)
        self.z2 = np.float64(0.0)

# ==============================================================================
# VocalSynth: Lillith's Internal Vocal Apparatus
# ==============================================================================

class VocalSynth:
    """
    Lillith's Internal Vocal Apparatus: Pure NumPy implementation of a simplified
    source-filter model for speech synthesis, using np.float64 precision.
    """
    def __init__(self, sample_rate: int = 44100, buffer_size_samples: int = 4410): # 0.1 second buffer
        self.sample_rate = sample_rate
        self.buffer_size_samples = buffer_size_samples
        self.current_time = 0.0 # Tracks time for waveform generation
        
        # --- Default/Initial Synthesis Parameters (will be controlled by Output.py) ---
        self.params = {
            # Pitch & Timing (Source Control)
            'F0_hz': np.float64(120.0),          # Fundamental Frequency (pitch)
            'Vibrato_Rate': np.float64(5.0),     # Hz
            'Vibrato_Extent': np.float64(2.0),   # Hz deviation
            # Loudness & Voice Quality (Source & Mix Control)
            'Amplitude': np.float64(0.5),        # Overall loudness (0-1)
            'Voicing_Mix': np.float64(1.0),      # 1.0 = fully voiced, 0.0 = fully unvoiced
            'Spectral_Tilt': np.float64(0.0),    # Affects brightness (e.g., -1.0 to 1.0)
            'Breathiness_Amount': np.float64(0.0), # (0-1)
            # Vocal Tract Shape (Filter Control - Formants)
            'F1_hz': np.float64(700.0),          # Formant 1 frequency (e.g., for 'ah' vowel)
            'F2_hz': np.float64(1200.0),         # Formant 2 frequency
            'F3_hz': np.float64(2500.0),         # Formant 3 frequency
            'F4_hz': np.float64(3500.0),         # Formant 4 frequency
            'BW1_hz': np.float64(60.0),          # Formant 1 bandwidth
            'BW2_hz': np.float64(90.0),          # Formant 2 bandwidth
            'BW3_hz': np.float64(120.0),         # Formant 3 bandwidth
            'BW4_hz': np.float64(150.0),         # Formant 4 bandwidth
            # Dynamic Expression (Overall Shaping)
            'Tension_Param': np.float64(0.0),    # (e.g., -1.0 to 1.0)
            'Roughness_Param': np.float64(0.0),  # (0-1)
        }
        # Ensure all parameters are float64
        for k in self.params:
            self.params[k] = np.float64(self.params[k])

        # --- Internal State for Source Model ---
        self._phase = np.float64(0.0)
        self._last_f0 = self.params['F0_hz'] # For smooth pitch transitions
        self._glottal_source_state = np.float64(0.0) # For simple pulse train

        # --- Internal State for Filter Model (Formant Filters) ---
        self.formant_filters = []
        self._initialize_formant_filters()
        
        logger.info(f"VocalSynth initialized with sample_rate={sample_rate}, buffer_size={buffer_size_samples}.")
        logger.info(f"VocalSynth using np.float64 precision for all calculations.")

    def _initialize_formant_filters(self):
        """Initializes/updates the biquad filters for formants."""
        self.formant_filters = []
        formant_indices = [1, 2, 3, 4] # For F1, F2, F3, F4
        
        for i in formant_indices:
            F = self.params[f'F{i}_hz']
            BW = self.params[f'BW{i}_hz']
            
            # Calculate filter coefficients for a resonant peak (biquad filter)
            # Based on standard digital filter design for resonant peaks (e.g., peaking EQ)
            # Or simpler direct formants
            
            # Pole-zero filter coefficients for a single formant
            # r = np.exp(-np.pi * BW / self.sample_rate) # Radius
            # theta = 2 * np.pi * F / self.sample_rate   # Angle
            # a1 = -2 * r * np.cos(theta)
            # a2 = r * r
            # b0 = 1.0
            # b1 = 0.0
            # b2 = 0.0
            
            # For a pure resonant filter, more commonly:
            R = np.exp(-np.pi * BW / self.sample_rate)
            omega = 2 * np.pi * F / self.sample_rate
            
            a1 = -2 * R * np.cos(omega)
            a2 = R**2
            
            # Numerator coefficients for a resonant filter, can be simplified to 1 for basic
            # For a gain-compensated peak, g = 1, we can use:
            b0 = (1 - R**2) # Simplified for gain, ensures 0dB peak
            b1 = 0.0
            b2 = 0.0

            # Ensure a0 is 1.0 for biquad standard form (y[n] = b0*x[n] + ... - a1*y[n-1]...)
            a0 = 1.0 
            
            b_coeffs = np.array([b0, b1, b2], dtype=np.float64)
            a_coeffs = np.array([a0, a1, a2], dtype=np.float64)

            self.formant_filters.append(BiquadFilter(b_coeffs, a_coeffs))
            # Reset filter state on update
            self.formant_filters[-1].reset_state()

    def _generate_glottal_source_sample(self, current_f0: np.float64) -> np.float64:
        """
        Generates a single sample of the glottal source signal.
        Uses a simple impulse train / modified sawtooth for voiced, and noise for unvoiced.
        """
        sample_period = 1.0 / self.sample_rate
        
        # Voiced Source (approximated as a pulse train/modified sawtooth)
        # _phase cycles from 0 to 1 at current_f0 rate
        self._phase += current_f0 * sample_period
        self._phase = self._phase % 1.0 # Keep phase within 0-1

        # Simple pulse train: impulse at phase 0, then decay.
        # More realistic: Liljencrants-Fant (LF) model approximation.
        # For lean implementation, let's use a rectified sine wave or simple impulse.
        # Here, a simple impulse, followed by an exponential decay for a crude 'glottal' shape.
        
        # When phase crosses 0 (or 1), a new pulse starts.
        if self._phase < sample_period * current_f0: # Small window around the "start" of a period
            self._glottal_source_state = np.float64(10.0) # Stronger impulse (was 1.0)
        else:
            # Exponential decay - slower decay for better sustain
            decay_factor = np.exp(- (current_f0 / 2000.0) ) # Slower decay (was /500.0)
            self._glottal_source_state *= decay_factor
        
        voiced_source = self._glottal_source_state
        
        # Unvoiced Source (white noise)
        unvoiced_source = (np.random.rand() * 2.0 - 1.0) # -1.0 to 1.0

        # Mix voiced and unvoiced based on Voicing_Mix parameter
        mixed_source = (voiced_source * self.params['Voicing_Mix'] + 
                        unvoiced_source * (1.0 - self.params['Voicing_Mix']))
        
        # Add breathiness (just more noise, before filtering)
        mixed_source += (np.random.rand() * 2.0 - 1.0) * self.params['Breathiness_Amount'] * 0.1 # Scaled breathiness
        
        return mixed_source.astype(np.float64)

    def generate_audio_buffer(self, control_params: Dict[str, np.float64]) -> np.ndarray:
        """
        Generates a buffer of audio samples based on the current control parameters.
        Control params are expected to be from Output.py.
        """
        # Update current synthesis parameters
        for param, value in control_params.items():
            if param in self.params:
                self.params[param] = np.float64(value)
            else:
                logger.warning(f"VocalSynth: Unknown control parameter '{param}'. Ignoring.")

        # Ensure formants are updated if frequencies/bandwidths changed
        # This check should be more robust, comparing new vs old params
        self._initialize_formant_filters() 

        audio_buffer = np.zeros(self.buffer_size_samples, dtype=np.float64)

        for i in range(self.buffer_size_samples):
            # Apply vibrato to F0
            current_f0 = self.params['F0_hz'] + (self.params['Vibrato_Extent'] * 
                                                  np.sin(2 * np.pi * self.params['Vibrato_Rate'] * self.current_time))
            current_f0 = np.clip(current_f0, 20.0, 1000.0) # Constrain F0 to reasonable range

            # Generate raw glottal source sample
            source_sample = self._generate_glottal_source_sample(current_f0)

            # Apply Spectral Tilt (simple low-pass/high-pass filter based on parameter)
            # A positive tilt makes it brighter (boost highs), negative makes it darker (boost lows).
            # For simplicity:
            # This is a conceptual placeholder; actual implementation uses filtering or direct source shaping.
            # A direct way would be to just multiply with some frequency dependent response.
            # For now, it will apply a slight gain to the source sample based on its value and the tilt.
            tilt_factor = 1.0 + (source_sample * self.params['Spectral_Tilt'] * 0.1) # Crude application
            filtered_source = source_sample * tilt_factor

            # Apply Vocal Tract Filter (Formant filters in cascade)
            filtered_output = filtered_source
            for formant_filter in self.formant_filters:
                filtered_output = formant_filter.process_sample(filtered_output)
            
            # Apply gain compensation for formant filter attenuation
            # Since we have 4 filters in cascade, each with gain < 1, compensate
            gain_compensation = 4.0  # Boost to compensate for cascade attenuation
            filtered_output *= gain_compensation

            # Apply Lip/Mouth Radiation (simple high-pass filter for clarity)
            # This is commonly a first-order differentiator or high-pass filter.
            # For simplicity, we can conceptualize it as adding a high-frequency bias.
            # No explicit filter implemented here to keep code leaner.

            # Overall amplitude scaling
            final_sample = filtered_output * self.params['Amplitude']

            audio_buffer[i] = final_sample
            self.current_time += 1.0 / self.sample_rate # Advance time

        # Ensure output is within -1.0 to 1.0 range
        audio_buffer = np.clip(audio_buffer, -1.0, 1.0)
        return audio_buffer.astype(np.float64) # Ensure float64 output

    def set_parameters(self, params: Dict[str, np.float64]):
        """Directly sets synthesis parameters."""
        for k, v in params.items():
            if k in self.params:
                self.params[k] = np.float64(v)
            else:
                logger.warning(f"VocalSynth: Attempted to set unknown parameter '{k}'.")
        self._initialize_formant_filters() # Re-initialize filters if formants changed

    def get_parameters(self) -> Dict[str, np.float64]:
        """Returns current synthesis parameters."""
        return self.params.copy()

    # --- Backward Compatibility Layer ---
    def synthesize_speech(self, language_vector: np.ndarray) -> np.ndarray:
        """Legacy interface expected by main.py.

        Takes an internal language activation vector (ignored for now) and
        generates an audio buffer using current control parameters.
        """
        try:
            control_params = self.get_parameters()
            return self.generate_audio_buffer(control_params)
        except Exception as e:
            logger.error(f"VocalSynth.synthesize_speech error: {e}")
            return np.zeros(self.buffer_size_samples, dtype=np.float64)

    # Persistence methods (save/load state)
    # def save_state(self, save_path: str):
    #     """Saves the VocalSynth's state to the single document."""
    #     try:
    #         state_data = {
    #             'params': {k: v.tolist() for k, v in self.params.items()},
    #             '_phase': self._phase.tolist(),
    #             '_last_f0': self._last_f0.tolist(),
    #             '_glottal_source_state': self._glottal_source_state.tolist(),
    #             'current_time': self.current_time,
    #             'filter_states': [(f.z1.tolist(), f.z2.tolist()) for f in self.formant_filters]
    #         }
    #         # Load existing global state
    #         if GLOBAL_STATE_FILE.exists():
    #             with open(GLOBAL_STATE_FILE, 'rb') as f:
    #                 global_state = pickle.load(f)
    #         else:
    #             global_state = {"version": 1, "modules": {}}
    #         # Add vocalsynth state
    #         global_state["modules"]["vocalsynth"] = state_data
    #         # Save back
    #         with open(GLOBAL_STATE_FILE, 'wb') as f:
    #             pickle.dump(global_state, f)
    #         logger.info(f"VocalSynth state saved to single document.")
    #     except Exception as e:
    #         logger.error(f"Error saving VocalSynth state: {e}")

    # def load_state(self, load_path: str):
    #     """Loads the VocalSynth's state from the single document."""
    #     try:
    #         if GLOBAL_STATE_FILE.exists():
    #             with open(GLOBAL_STATE_FILE, 'rb') as f:
    #                 global_state = pickle.load(f)
    #             loaded_state = global_state.get("modules", {}).get("vocalsynth")
    #             if loaded_state:
    #                 for k, v_list in loaded_state['params'].items():
    #                     self.params[k] = np.float64(v_list)
    #                 self._phase = np.float64(loaded_state['_phase'])
    #                 self._last_f0 = np.float64(loaded_state['_last_f0'])
    #                 self._glottal_source_state = np.float64(loaded_state['_glottal_source_state'])
    #                 self.current_time = loaded_state['current_time']
    #                 # Filter states would be restored here
    #                 logger.info(f"VocalSynth state loaded from single document.")
    #             else:
    #                 logger.warning("Vocalsynth state not found in single document. Using defaults.")
    #                 self.params = {k: np.float64(v) for k, v in self.params.items()}
    #                 self._initialize_formant_filters()
    #         else:
    #             logger.warning(f"Single document not found. Using defaults.")
    #             self.params = {k: np.float64(v) for k, v in self.params.items()}
    #             self._initialize_formant_filters()
    #     except Exception as e:
    #         logger.error(f"Error loading VocalSynth state: {e}. Using defaults.")
    #         self.params = {k: np.float64(v) for k, v in self.params.items()}
    #         self._initialize_formant_filters()


# Test block (can be removed in final deployment)
if __name__ == "__main__":
    logger.info("Running VocalSynth test.")

    synth = VocalSynth(sample_rate=44100, buffer_size_samples=44100 // 10) # 0.1 second buffer

    # --- Test 1: Generate a simple vowel sound ('ah') ---
    logger.info("\n--- Test 1: Generating a simple vowel sound ('ah') ---")
    
    # Set parameters for a basic 'ah' vowel
    ah_params = synth.get_parameters()
    ah_params['F0_hz'] = np.float64(150.0)
    ah_params['Amplitude'] = np.float64(0.6)
    ah_params['Voicing_Mix'] = np.float64(1.0)
    ah_params['F1_hz'] = np.float64(700.0)
    ah_params['F2_hz'] = np.float64(1200.0)
    ah_params['F3_hz'] = np.float64(2500.0)
    ah_params['F4_hz'] = np.float64(3500.0)
    ah_params['Spectral_Tilt'] = np.float64(-0.5) # Slightly darker

    audio_buffer_ah = synth.generate_audio_buffer(ah_params)
    logger.info(f"Generated 'ah' vowel buffer shape: {audio_buffer_ah.shape}, Max Amp: {np.max(np.abs(audio_buffer_ah)):.4f}")
    # You would typically play this audio here (e.g., using sounddevice)

    # --- Test 2: Generate an unvoiced sound (hiss) ---
    logger.info("\n--- Test 2: Generating an unvoiced sound (hiss) ---")
    hiss_params = synth.get_parameters()
    hiss_params['Amplitude'] = np.float64(0.4)
    hiss_params['Voicing_Mix'] = np.float64(0.0) # Fully unvoiced
    hiss_params['Breathiness_Amount'] = np.float64(0.8) # More breathiness
    hiss_params['F0_hz'] = np.float64(0.0) # F0 irrelevant for unvoiced

    audio_buffer_hiss = synth.generate_audio_buffer(hiss_params)
    logger.info(f"Generated 'hiss' buffer shape: {audio_buffer_hiss.shape}, Max Amp: {np.max(np.abs(audio_buffer_hiss)):.4f}")

    # --- Test 3: Generate a sound with vibrato ---
    logger.info("\n--- Test 3: Generating sound with vibrato ---")
    vib_params = synth.get_parameters()
    vib_params['F0_hz'] = np.float64(180.0)
    vib_params['Amplitude'] = np.float64(0.5)
    vib_params['Voicing_Mix'] = np.float64(1.0)
    vib_params['Vibrato_Rate'] = np.float64(6.0) # Faster vibrato
    vib_params['Vibrato_Extent'] = np.float64(5.0) # Wider vibrato
    
    audio_buffer_vib = synth.generate_audio_buffer(vib_params)
    logger.info(f"Generated vibrato buffer shape: {audio_buffer_vib.shape}, Max Amp: {np.max(np.abs(audio_buffer_vib)):.4f}")


    # --- Test Save/Load ---
    logger.info("\n--- Testing Save/Load ---")
    # Prepare state data
    vocalsynth_state = {
        'params': {k: v.tolist() for k, v in synth.params.items()},
        '_phase': synth._phase.tolist(),
        '_last_f0': synth._last_f0.tolist(),
        '_glottal_source_state': synth._glottal_source_state.tolist(),
        'current_time': synth.current_time,
        # Filter states (z1, z2) need to be saved for each BiquadFilter instance
        'filter_states': [(f.z1.tolist(), f.z2.tolist()) for f in synth.formant_filters]
    }

    # Save state using pickle
    with open(GLOBAL_STATE_FILE, 'wb') as f:
        pickle.dump(vocalsynth_state, f)

    # Load test
    with open(GLOBAL_STATE_FILE, 'rb') as f:
        loaded_state = pickle.load(f)
    if loaded_state:
        # Apply loaded state to new synth object
        new_synth = VocalSynth(sample_rate=44100, buffer_size_samples=44100 // 10)
        for k, v_list in loaded_state['params'].items():
            new_synth.params[k] = np.float64(v_list)
        
        new_synth._phase = np.float64(loaded_state['_phase'])
        new_synth._last_f0 = np.float64(loaded_state['_last_f0'])
        new_synth._glottal_source_state = np.float64(loaded_state['_glottal_source_state'])
        new_synth.current_time = loaded_state['current_time']
        # Filter states would be restored here
    logger.info(f"VocalSynth state loaded.")


    logger.info("VocalSynth test complete.")