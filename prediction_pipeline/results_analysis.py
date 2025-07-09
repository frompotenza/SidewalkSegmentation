import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to experiments
base_dir = (
    r"C:\Users\poten\SidewalkSegmentation\YOLO_training\hyperparameter_tunning_results"
)
experiments = sorted(
    [d for d in os.listdir(base_dir) if d.startswith("seg_experiment")]
)

metrics = [
    "metrics/precision",
    "metrics/recall",
    "metrics/mAP_0.5",
    "metrics/mAP_0.5:0.95",
]
final_results = []

for exp in experiments:
    csv_path = os.path.join(base_dir, exp, "results.csv")
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    final_epoch = df.iloc[-1]
    results = {
        "experiment": exp,
        "lr0": exp.split("lr0=")[1].split(",")[0],
        "wd": exp.split("weight_decay=")[1].split(",")[0],
        "batch": exp.split("batch=")[1],
    }
    for m in metrics:
        results[m] = final_epoch.get(m, None)
    final_results.append(results)

# Convert to DataFrame
results_df = pd.DataFrame(final_results)

# Plot
plt.figure(figsize=(12, 6))
for metric in metrics:
    plt.plot(results_df["experiment"], results_df[metric], marker="o", label=metric)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Metric Value")
plt.title("YOLO Segmentation Model Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
