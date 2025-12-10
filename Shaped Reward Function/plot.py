import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os
import threading
import time
import argparse

# ------------------------
# CONFIGURATION
# ------------------------

plt.style.use(["dark_background", "seaborn-v0_8-dark-palette"])
SMALL_SIZE = 10
plt.rcParams.update({
    "font.size": SMALL_SIZE,
    "axes.titlesize": SMALL_SIZE,
    "axes.labelsize": SMALL_SIZE,
    "xtick.labelsize": SMALL_SIZE,
    "ytick.labelsize": SMALL_SIZE,
    "legend.fontsize": SMALL_SIZE,
    "figure.titlesize": SMALL_SIZE
})

# ------------------------
# PARSE ARGUMENTS
# ------------------------
parser = argparse.ArgumentParser(description="Lunar Lander metrics plotter")
parser.add_argument("--file", type=str, default="lunarlanderMetric.csv", help="Path to the metrics CSV file")
parser.add_argument("--interval", type=int, default=10, help="Update interval in seconds")
args = parser.parse_args()

AGENT_TRAIN_METRIC_PATH = args.file
UPDATE_INTERVAL = args.interval

# ------------------------
# GLOBAL DATA + LOCK
# ------------------------
episodic_data = pd.DataFrame()
data_lock = threading.Lock()

# ------------------------
# DATA READER THREAD
# ------------------------
def read_data():
    global episodic_data
    while True:
        try:
            if os.path.exists(AGENT_TRAIN_METRIC_PATH):
                new_data = pd.read_csv(AGENT_TRAIN_METRIC_PATH)
                with data_lock:
                    episodic_data = new_data.copy()
            else:
                print(f"[WARN] File not found: {AGENT_TRAIN_METRIC_PATH}")
        except Exception as e:
            print(f"[ERROR] Reading data failed: {e}")
        time.sleep(UPDATE_INTERVAL)

thread = threading.Thread(target=read_data, daemon=True)
thread.start()

# ------------------------
# SETUP FIGURE + SUBPLOTS
# ------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Lunar Lander Training Metrics", fontsize=14)
(ax1, ax2), (ax3, ax4) = axes

# ------------------------
# PLOTTING FUNCTION
# ------------------------
def update(_):
    with data_lock:
        if episodic_data.empty:
            return

        x = episodic_data["episode"]

        ax1.cla()
        ax1.plot(x, episodic_data["totalReward"], color="#2AD4FF", linewidth=1)
        ax1.set_title("Total Reward per Episode")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")

        ax2.cla()
        ax2.plot(x, episodic_data["avgQ"], color="#FF00FF", linewidth=1)
        ax2.set_title("Average Q Value")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Q Value")

        ax3.cla()
        ax3.plot(x, episodic_data["length"], color="#00FF00", linewidth=1)
        ax3.set_title("Episode Length")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Length")

        ax4.cla()
        ax4.plot(x, episodic_data["exploration"], color="#FF6600", linewidth=1)
        ax4.set_title("Exploration (Epsilon)")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Epsilon")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

# ------------------------
# ANIMATION
# ------------------------
ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL * 1000)
plt.show()