# c:\ace3\tokenizerout.py

import time
import os

class TokenizerOut:
    """Receives output tokens from the framework and logs them."""
    def __init__(self, log_dir="c:/ace3/data_collection"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        session_id = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"expressive_impulses_{session_id}.log")
        print(f"📝 TokenizerOut will log expressive impulses to {self.log_path}")

    def receive_tokens(self, tokens):
        """Logs the received tokens with a timestamp."""
        if tokens is None or tokens.numel() == 0:
            return
        
        token_list = tokens.cpu().numpy().tolist()
        log_entry = f"{time.time()}: {token_list}\n"
        
        with open(self.log_path, 'a') as f:
            f.write(log_entry)