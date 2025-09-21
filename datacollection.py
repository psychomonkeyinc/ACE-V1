# c:\ace3\datacollection.py

import os
import time
import json
import torch
from enum import Enum  # NEW: Import the Enum library to check for its type

class DataCollection:
    """Handles data logging with periodic checkpointing."""
    def __init__(self, base_dir="c:/ace3/data_collection", checkpoint_cycle=30):
        self.session_id = time.strftime("%Y%m%d-%H%M%S")
        self.session_dir = os.path.join(base_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.log_path = os.path.join(self.session_dir, "consciousness_stream.jsonl")
        print(f"📊 DataCollection initialized. Logging to: {self.log_path}")
        
        self.checkpoint_cycle = checkpoint_cycle
        self.cycle_count = 0
        self.buffer = []
        
        self.cycle_start_time = time.perf_counter()

    def capture_snapshot(self, framework_obj, cycle_results, original_timestamp):
        """Buffers a snapshot and calculates latency before writing."""
        cycle_end_time = time.perf_counter()
        
        serializable_results = self._tensor_to_list(cycle_results)
        
        latency_ms = (time.time() - original_timestamp) * 1000 if original_timestamp else None
        
        snapshot = {
            "timestamp": time.time(),
            "input_timestamp": original_timestamp,
            "cycle_duration_ms": (cycle_end_time - self.cycle_start_time) * 1000,
            "end_to_end_latency_ms": latency_ms,
            "results": serializable_results
        }
        
        self.buffer.append(snapshot)
        self.cycle_count += 1
        
        if self.cycle_count >= self.checkpoint_cycle:
            self.flush_buffer()
            
        self.cycle_start_time = cycle_end_time
        return snapshot

    def flush_buffer(self):
        """Writes the contents of the buffer to the log file and clears it."""
        if not self.buffer:
            return
        with open(self.log_path, 'a') as f:
            for item in self.buffer:
                json.dump(item, f)
                f.write('\n')
        print(f"\nSYSTEM: Data checkpoint saved ({len(self.buffer)} cycles).")
        self.buffer = []
        self.cycle_count = 0

    def _tensor_to_list(self, item):
        """
        Recursively converts special objects (Tensors, Enums) to lists/strings for JSON.
        """
        # NEW: Check if the item is an Enum and get its string value (e.g., "simple_words")
        if isinstance(item, Enum):
            return item.value
        
        if isinstance(item, torch.Tensor):
            return item.cpu().numpy().tolist()
        if isinstance(item, dict):
            return {k: self._tensor_to_list(v) for k, v in item.items()}
        if isinstance(item, list):
            return [self._tensor_to_list(i) for i in item]
        return item