# c:\ace3\dataviewer.py or c:\acelearn\dataviewer.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def find_latest_session(base_dir):
    """Finds the most recently created session folder in the specified directory."""
    try:
        session_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not session_folders:
            return None
        latest_folder = max(session_folders, key=os.path.getctime)
        return latest_folder
    except FileNotFoundError:
        return None

def load_data(session_folder):
    """Loads all cycle data from the session's log file."""
    log_file = os.path.join(session_folder, "consciousness_stream.jsonl")
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return []
    
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse a line in the log file: {line}")
    return data

def print_summary(data, session_folder):
    """Prints a simple summary of the session."""
    if not data:
        return
        
    total_cycles = len(data)
    start_time = data[0]['timestamp']
    end_time = data[-1]['timestamp']
    duration_seconds = end_time - start_time
    
    cycle_times = [d['cycle_duration_ms'] for d in data if d.get('cycle_duration_ms') is not None]
    avg_cycle_ms = sum(cycle_times) / len(cycle_times) if cycle_times else 0

    print("\n--- Session Summary ---")
    print(f"Session Folder: {os.path.basename(session_folder)}")
    print(f"Total Duration: {duration_seconds:.2f} seconds")
    print(f"Total Cycles Logged: {total_cycles}")
    print(f"Average Cycle Time: {avg_cycle_ms:.2f} ms")
    print("-----------------------\n")

def plot_full_diagnostics(data):
    """Generates a full 16-panel plot for all major data streams."""
    if not data:
        print("No data to plot.")
        return

    timestamps = np.array([d['timestamp'] for d in data])
    timestamps -= timestamps[0]

    # --- EXTRACT ALL DATA STREAMS ---
    # We use safe .get() calls with defaults to prevent crashes if a key is missing.
    
    # Simple scalar values
    health_scores = [d.get('results', {}).get('health', {}).get('health_score', [[0]])[0][0] for d in data]
    security_scores = [d.get('results', {}).get('security', {}).get('security_score', [[0]])[0][0] for d in data]
    memory_relevance = [d.get('results', {}).get('memory', {}).get('memory_relevance', [[0]])[0] for d in data]
    output_confidence = [d.get('results', {}).get('output', {}).get('confidence', [[0]])[0] for d in data]
    awareness_levels = [d.get('results', {}).get('conscience', {}).get('awareness_level', [[0]])[0][0] for d in data]
    bonding_strengths = [d.get('results', {}).get('attachment_style', {}).get('bonding_strength', [[0]])[0][0] for d in data]
    
    # Vector magnitudes (intensity levels)
    perception_intensity = [np.linalg.norm(d.get('results',{}).get('perception',{}).get('fused_features',[0])) for d in data]
    emotion_intensity = [np.linalg.norm(d.get('results',{}).get('emotion_simulation',{}).get('emotional_state',[0])) for d in data]
    cognitive_load = [np.linalg.norm(d.get('results',{}).get('cognitive',{}).get('unified_cognitive_state',[0])) for d in data]
    vocal_effort = [np.linalg.norm(d.get('results',{}).get('vocal',{}).get('shaped_vocalization',[0])) for d in data]
    language_activity = [np.linalg.norm(d.get('results',{}).get('language',{}).get('global_repr',[0])) for d in data]
    tom_intensity = [np.linalg.norm(d.get('results',{}).get('theory_of_mind',{}).get('empathic_understanding',{}).get('empathy_features',[0])) for d in data]
    emem_intensity = [np.linalg.norm(d.get('results',{}).get('emotional_memory',{}).get('emotional_memory',[0])) for d in data]
    learning_intensity = [np.linalg.norm(d.get('results',{}).get('learning',{}).get('learning_output',[0])) for d in data]
    
    # Scatter plot data
    output_tokens = [d.get('results', {}).get('output', {}).get('response_tokens', [0])[0] for d in data]

    # --- Create the 4x4 Plot Grid ---
    fig, axs = plt.subplots(4, 4, figsize=(24, 18), facecolor='#f0f0f0')
    fig.suptitle('ACE Full Diagnostic Panel', fontsize=20)
    
    plot_map = {
        (0, 0): (health_scores, 'System Health', 'lime'),
        (0, 1): (security_scores, 'System Security', 'red'),
        (0, 2): (awareness_levels, 'Awareness Level', 'cyan'),
        (0, 3): (cognitive_load, 'Cognitive Load', 'orange'),
        
        (1, 0): (perception_intensity, 'Perception Intensity', 'lightblue'),
        (1, 1): (emotion_intensity, 'Emotion Intensity', 'magenta'),
        (1, 2): (memory_relevance, 'Memory Relevance', 'blue'),
        (1, 3): (emem_intensity, 'Emotional Memory Intensity', 'purple'),

        (2, 0): (tom_intensity, 'Theory of Mind Intensity', 'yellow'),
        (2, 1): (bonding_strengths, 'Attachment Bonding Strength', 'pink'),
        (2, 2): (language_activity, 'Internal Language Activity', 'white'),
        (2, 3): (learning_intensity, 'Learning Signal Intensity', 'gray'),

        (3, 0): (output_confidence, 'Output Confidence', 'teal'),
        (3, 1): (vocal_effort, 'Vocal Effort', 'brown'),
    }

    for (r, c), (p_data, title, color) in plot_map.items():
        axs[r, c].plot(timestamps, p_data, color=color)
        axs[r, c].set_title(title)
        axs[r, c].grid(True, linestyle='--', alpha=0.4)
        if r == 3: axs[r,c].set_xlabel('Time (seconds)')

    # Special scatter plot for tokens
    axs[3, 2].plot(timestamps, output_tokens, color='gold', marker='.', linestyle='None')
    axs[3, 2].set_title('Primary Expressive Token')
    axs[3, 2].set_xlabel('Time (seconds)')
    axs[3, 2].grid(True, linestyle='--', alpha=0.4)
    
    # Hide unused plots
    axs[3, 3].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    print("Starting ACE Full Diagnostic Data Viewer...")
    # Allow user to choose which instance's data to view
    instance_choice = input("View data for which instance? (ace3 / acelearn) [default: acelearn]: ")
    if instance_choice.lower() == 'ace3':
        data_dir = "c:/ace3/data_collection"
    else:
        data_dir = "c:/acelearn/data_collection"
        
    print(f"Looking for data in: {data_dir}")
    latest_session = find_latest_session(data_dir)
    
    if latest_session:
        print(f"Found latest session: {os.path.basename(latest_session)}")
        session_data = load_data(latest_session)
        if session_data:
            print_summary(session_data, latest_session)
            plot_full_diagnostics(session_data)
        else:
            print("Session data file is empty or could not be read.")
    else:
        print(f"No session data found in '{data_dir}'.")