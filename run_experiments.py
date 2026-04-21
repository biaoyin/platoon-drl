import os
import subprocess
import itertools
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import csv

from env.config import ENV_CONFIG, TRAIN_CONFIG

algo = "DoubleDQNAgent_lr0.0002"
flow_values = [500, 1000, 1500, 2000]

FLOW_DIR = "config/run_experiments"
LOG_DIR = "logs/test/acc_actions"
SUMO_CFG_DIR = FLOW_DIR
RESULT_FILE = "results.csv"

template = """<additional> 
    <flow id="m" begin="0" end="1000000" departPos="base" departSpeed="max" departLane="0" vehsPerHour="{flow_m}" type="vmix" route="m" /> 
    <flow id="p_m" begin="0" end="1000000" departPos="base" departSpeed="max" departLane="1" vehsPerHour="{flow_pm}" type="cav2" route="m" /> 
</additional> 
"""

SUMO_CFG_TEMPLATE = """<configuration>
    <input>
        <net-file value="my_net.net.xml"/>
        <additional-files value="old_net.rou.xml,{flow_file},detectors.add.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="1000000"/>
    </time>
</configuration>
"""

def compute_avg(log_dir, tag="AvgCol"):
    try:
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={'scalars': 0}
        )
        ea.Reload()

        if tag not in ea.Tags()["scalars"]:
            print(f"[WARNING] Tag '{tag}' not found in {log_dir}")
            return None

        scalars = ea.Scalars(tag)
        values = [s.value for s in scalars]

        return np.mean(values) if values else None

    except Exception as e:
        print(f"[ERROR] Failed reading {log_dir}: {e}")
        return None


# -----------------------------
# CSV init
# -----------------------------
if not os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["flow_m", "flow_pm", "avg_collision"])

# Main loop (16 runs)
for flow_m, flow_pm in itertools.product(flow_values, flow_values):

    flow_name = f"m{flow_m}_pm{flow_pm}"
    flow_file = os.path.join(FLOW_DIR, f"{flow_name}.flow.xml")
    log_dir = os.path.join(LOG_DIR, flow_name)
    cfg_file = os.path.join(SUMO_CFG_DIR, f"run_{flow_name}.sumocfg")

    #os.makedirs(log_dir, exist_ok=True)

    # Create flow XML
    with open(flow_file, "w") as f:
        f.write(template.format(flow_m=flow_m, flow_pm=flow_pm))


    # Create config
    with open(cfg_file, "w") as f:
        f.write(SUMO_CFG_TEMPLATE.format(flow_file=f"{flow_name}.flow.xml"))

    print(f"Running: m={flow_m}, p_m={flow_pm}")

    subprocess.run([
        "python", "test.py",
        "-conf", cfg_file,
        "-log_dir", log_dir
    ])
    print(f"Finished: m={flow_m}, p_m={flow_pm}")

    avg_collision = compute_avg(f"{log_dir+algo}")
    with open("results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([flow_m, flow_pm, avg_collision])
    print(f"Average collision saved: {avg_collision}")


