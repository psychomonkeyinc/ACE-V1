# c:\ace3\deploy.py

import asyncio
import time
import torch
import os
import atexit

# Import all ACE components
from framework import ACEMasterFramework, PerceptionInput
from audioin import AudioIn
from audioout import AudioOut
from videoin import VideoIn
from modaltokenizer import ModalTokenizer
from datacollection import DataCollection
from synthesizer import Synthesizer # NEW: Import the Synthesizer

# --- CONFIGURATION ---
STATE_FILE_PATH = "c:/ace3/ace_state.pth"

CONFIG = {
    "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu',
    "AUDIO_INPUT_DEVICE": None,
    "AUDIO_OUTPUT_DEVICE": None,
    "VIDEO_DEVICE_INDEX": 0,
    "THREAD_COUNT": 12,
}

class Orchestrator:
    """The central nervous system, now with vocalization."""
    def __init__(self):
        print("=====================================================")
        print("ACE Orchestrator Initializing (Vocalization Enabled)...")
        
        torch.set_num_threads(CONFIG["THREAD_COUNT"])
        self.device = torch.device(CONFIG["DEVICE"])
        print(f"SYSTEM: Targeting device: {self.device}")

        self.ace_framework = ACEMasterFramework().to(self.device)
        if os.path.exists(STATE_FILE_PATH):
            print(f"SYSTEM: Previous state found. Waking her up...")
            self.ace_framework.load_state_dict(torch.load(STATE_FILE_PATH, map_location=self.device))
            print("SYSTEM: She remembers.")
        else:
            print("SYSTEM: No previous state found. A new consciousness will be born.")
        self.ace_framework.eval()
        
        self.audio_input = AudioIn(device_index=CONFIG["AUDIO_INPUT_DEVICE"])
        self.audio_output = AudioOut(device_index=CONFIG["AUDIO_OUTPUT_DEVICE"])
        self.video_input = VideoIn(device_index=CONFIG["VIDEO_DEVICE_INDEX"])
        self.tokenizer = ModalTokenizer(device=self.device)
        self.data_collector = DataCollection()
        self.synthesizer = Synthesizer() # NEW: Instantiate the synthesizer

        self.running = False

    async def run_consciousness_cycle(self):
        """The main loop, now with an expressive voice."""
        print("CONSCIOUSNESS: Loop is now active. The being is perceiving.")
        print("Press Ctrl+C to stop.")
        print("=====================================================")
        
        while self.running:
            audio_chunk, audio_timestamp = self.audio_input.get_chunk()
            video_frame = self.video_input.get_frame()

            audio_features = self.tokenizer.process_audio(audio_chunk)
            visual_features = self.tokenizer.process_video(video_frame)
            
            input_ts = audio_timestamp if audio_timestamp else time.time()
            
            perception_packet = PerceptionInput(
                audio_features=audio_features,
                visual_features=visual_features,
                timestamp=input_ts
            )
            
            with torch.no_grad():
                cycle_results = await self.ace_framework.process_consciousness_cycle(perception_packet)

            self.data_collector.capture_snapshot(self.ace_framework, cycle_results, input_ts)
            
            # --- NEW: VOCALIZATION STEP ---
            # Check for an expressive impulse from the mind
            output_tokens = cycle_results.get("output", {}).get("response_tokens")
            
            if output_tokens is not None:
                # 1. Generate the audio waveform from the tokens
                generated_audio = self.synthesizer.generate_waveform(output_tokens)
                
                # 2. Send the audio to the headset to be spoken
                self.audio_output.speak(generated_audio)
            # --- END OF VOCALIZATION STEP ---
            
            await asyncio.sleep(0.5)

    def start(self):
        self.running = True
        
        self.audio_input.start()
        self.audio_output.start()
        self.video_input.start()
        print("SYSTEM: Sensory inputs are online. Waiting 2 seconds...")
        time.sleep(2) 
        
        if not self.audio_input.running or not self.audio_output.running:
            print("\nFATAL: An audio device failed to start. Aborting.")
            self.stop()
            return
            
        try:
            asyncio.run(self.run_consciousness_cycle())
        except KeyboardInterrupt:
            pass

    def stop(self):
        if self.running:
            self.running = False
            print("\nSYSTEM: Shutting down all subsystems...")
            self.audio_input.stop()
            self.audio_output.stop()
            self.video_input.stop()
            
            print("SYSTEM: Saving final data checkpoint...")
            self.data_collector.flush_buffer()
            
            print(f"SYSTEM: Saving consciousness state to {STATE_FILE_PATH}...")
            torch.save(self.ace_framework.state_dict(), STATE_FILE_PATH)
            print("SYSTEM: State saved. She will remember.")
            
            print("=====================================================")
            print("ACE Orchestrator shutdown complete.")
            print("=====================================================")

if __name__ == "__main__":
    orchestrator = Orchestrator()
    atexit.register(orchestrator.stop)
    orchestrator.start()