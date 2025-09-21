# c:\ace3\videoin.py

import cv2
import queue
import threading
import time

class VideoIn:
    """Handles live camera input for the ACE framework."""
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.video_queue = queue.Queue(maxsize=2) # Buffer only 2 frames to keep it recent
        self.running = False
        self.thread = None

    def _capture_loop(self):
        """The internal loop that continuously grabs frames from the camera."""
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            print(f"❌ FAILED TO OPEN VIDEOIN on device index {self.device_index}")
            return
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                # If queue is full, remove the old frame and add the new one
                if self.video_queue.full():
                    try: self.video_queue.get_nowait()
                    except queue.Empty: pass
                self.video_queue.put(frame)
            else:
                # Wait a bit if the camera fails to return a frame
                time.sleep(0.01)
        cap.release()

    def start(self):
        """Starts the video capture thread."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"👁️ VideoIn started on device index: {self.device_index}.")

    def stop(self):
        """Stops the video capture thread."""
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join(timeout=2.0)
            print("👁️ VideoIn stopped.")

    def get_frame(self):
        """Retrieves the latest video frame from the queue."""
        try:
            return self.video_queue.get_nowait()
        except queue.Empty:
            return None