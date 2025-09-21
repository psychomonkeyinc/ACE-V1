# c:\ace3\audioin.py

import sounddevice as sd
import numpy as np
import queue
import threading

class AudioIn:
    """Handles INPUT-ONLY audio and adds a high-precision timestamp to each chunk."""
    def __init__(self, device_index=None, sample_rate=44100, chunk_duration=0.2):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.input_queue = queue.Queue(maxsize=10)
        self.running = False
        self.stream = None

    def _input_callback(self, indata, frames, time, status):
        """This function is called by the sounddevice stream for each audio block."""
        if status:
            print(f"AudioIn Warning: {status}")
        if self.running and not self.input_queue.full():
            # MODIFIED: Instead of just the data, we now queue a tuple containing
            # the data and the exact time the audio was captured by the hardware (ADC time).
            # This is the most accurate input timestamp we can get.
            self.input_queue.put((indata.copy(), time.inputBufferAdcTime))

    def start(self):
        """Starts the input audio stream."""
        try:
            self.running = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.device_index,
                blocksize=self.chunk_size,
                callback=self._input_callback
            )
            self.stream.start()
            print(f"🎤 AudioIn Stream started on device index: {self.device_index or 'Default'}.")
        except Exception as e:
            print(f"❌ FAILED TO START AUDIO INPUT: {e}")
            self.running = False

    def stop(self):
        """Stops the audio stream gracefully."""
        if self.stream is not None:
            self.running = False
            self.stream.stop()
            self.stream.close()
            print("🎤 AudioIn Stream stopped.")

    def get_chunk(self):
        """Retrieves a tuple of (audio_chunk, timestamp) from the queue."""
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None, None # Return None for both if empty