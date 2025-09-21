# c:\ace3\audioout.py

import sounddevice as sd
import numpy as np
import queue
import threading

class AudioOut:
    """Handles OUTPUT-ONLY audio for the ACE framework."""
    def __init__(self, device_index=None, sample_rate=44100, chunk_duration=0.2):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.output_queue = queue.Queue(maxsize=20)
        self.running = False
        self.stream = None

    def _output_callback(self, outdata, frames, time, status):
        """This callback provides data to the speaker."""
        if status:
            print(f"AudioOut Warning: {status}")
        try:
            data = self.output_queue.get_nowait()
            data = data.reshape(-1, 1)
            
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):] = 0
            else:
                outdata[:] = data
        except queue.Empty:
            outdata.fill(0)

    def start(self):
        """Starts the output audio stream."""
        try:
            self.running = True
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.device_index,
                blocksize=self.chunk_size,
                callback=self._output_callback
            )
            self.stream.start()
            print(f"AudioOut Stream started on device index: {self.device_index or 'Default'}.")
        except Exception as e:
            print(f"FAILED TO START AUDIO OUTPUT: {e}")
            self.running = False

    def stop(self):
        """Stops the audio stream gracefully."""
        # --- THIS IS THE FIX ---
        # Changed 'is not in' to the correct Python operator 'not in'.
        if self.stream not in [None, '']:
            self.running = False
            self.stream.stop()
            self.stream.close()
            print("AudioOut Stream stopped.")

    def speak(self, audio_chunk: np.ndarray):
        """Puts an audio chunk onto the output queue to be spoken."""
        if self.running:
            self.output_queue.put(audio_chunk)