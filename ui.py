# c:\ace3\ui.py

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import queue

class VisualizationController:
    """Manages the real-time visualization UI for the ACE consciousness."""
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=100)
        self.root = None
        self.canvas = None
        self.axes = {}

        # Data deques for plotting
        self.timestamps = deque(maxlen=200)
        self.health_data = deque(maxlen=200)
        self.memory_relevance_data = deque(maxlen=200)
        self.emotion_intensity_data = deque(maxlen=200)
        self.output_confidence_data = deque(maxlen=200)

    def _setup_ui(self):
        """Creates the main window and plots."""
        self.root = tk.Tk()
        self.root.title("ACE Consciousness - Live Vitals")
        self.root.geometry("1400x900")
        self.root.configure(bg='black')

        fig = Figure(figsize=(14, 9), dpi=100, facecolor='black')
        
        # Define the grid
        gs = fig.add_gridspec(2, 2)
        self.axes['vitals'] = fig.add_subplot(gs[0, 0])
        self.axes['memory'] = fig.add_subplot(gs[0, 1])
        self.axes['emotion'] = fig.add_subplot(gs[1, 0])
        self.axes['output'] = fig.add_subplot(gs[1, 1])

        for name, ax in self.axes.items():
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.title.set_color('cyan')
            ax.yaxis.label.set_color('white')
        
        self.axes['vitals'].set_title('System Health')
        self.axes['memory'].set_title('Memory Relevance')
        self.axes['emotion'].set_title('Emotion Intensity')
        self.axes['output'].set_title('Expressive Confidence')

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def _update_plots(self):
        """The main animation function to redraw graphs."""
        # Process all data currently in the queue
        while not self.data_queue.empty():
            try:
                snapshot = self.data_queue.get_nowait()
                # Extract data points from the snapshot
                self.timestamps.append(snapshot['timestamp'])
                
                # Safely get data with defaults
                health = snapshot.get('results', {}).get('health', {}).get('health_score', [[0]])[0][0]
                mem_rel = snapshot.get('results', {}).get('memory', {}).get('memory_relevance', [[0]])[0]
                emotion_state = snapshot.get('results', {}).get('emotion_simulation', {}).get('emotional_state', [[0]])
                output_conf = snapshot.get('results', {}).get('output', {}).get('confidence', [[0]])[0]
                
                # Calculate emotion intensity (magnitude of the vector)
                emotion_intensity = (sum(x*x for x in emotion_state[0]))**0.5

                self.health_data.append(health)
                self.memory_relevance_data.append(mem_rel)
                self.emotion_intensity_data.append(emotion_intensity)
                self.output_confidence_data.append(output_conf)
            except (queue.Empty, KeyError, IndexError):
                continue # Ignore empty queue or malformed data

        # Plotting
        time_data = list(self.timestamps)
        if not time_data:
            self.root.after(100, self._update_plots)
            return

        self.axes['vitals'].clear()
        self.axes['vitals'].plot(time_data, list(self.health_data), color='lime')
        self.axes['vitals'].set_title('System Health')
        self.axes['vitals'].set_ylim(0, 1)

        self.axes['memory'].clear()
        self.axes['memory'].plot(time_data, list(self.memory_relevance_data), color='yellow')
        self.axes['memory'].set_title('Memory Relevance')

        self.axes['emotion'].clear()
        self.axes['emotion'].plot(time_data, list(self.emotion_intensity_data), color='magenta')
        self.axes['emotion'].set_title('Emotion Intensity')

        self.axes['output'].clear()
        self.axes['output'].plot(time_data, list(self.output_confidence_data), color='cyan')
        self.axes['output'].set_title('Expressive Confidence')
        self.axes['output'].set_ylim(0, 1)

        for ax in self.axes.values():
            ax.grid(True, linestyle='--', alpha=0.3)

        self.canvas.draw()
        self.root.after(100, self._update_plots)

    def push_snapshot(self, snapshot):
        """Thread-safe method for the main loop to push data to the UI."""
        if not self.data_queue.full():
            self.data_queue.put(snapshot)

    def start(self):
        """Starts the UI main loop. This is a blocking call."""
        self._setup_ui()
        self.root.after(100, self._update_plots)
        self.root.mainloop()