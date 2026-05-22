import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("eval_results.json") as f:
    results = json.load(f)

lead_times = sorted(results.keys(), key=lambda x: int(x.replace("h", "")))
hours = [int(lt.replace("h", "")) for lt in lead_times]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RMSE
for var in ["z", "t", "t2m", "u10", "v10"]:
    rmses = [results[lt][var]["rmse_mean"] for lt in lead_times]
    axes[0].plot(hours, rmses, "o-", label=var, markersize=4)
axes[0].set_xlabel("Lead Time (hours)")
axes[0].set_ylabel("Lat-weighted RMSE")
axes[0].set_title("ClimODE - RMSE vs Lead Time")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ACC
for var in ["z", "t", "t2m", "u10", "v10"]:
    accs = [results[lt][var]["acc_mean"] for lt in lead_times]
    axes[1].plot(hours, accs, "o-", label=var, markersize=4)
axes[1].set_xlabel("Lead Time (hours)")
axes[1].set_ylabel("ACC")
axes[1].set_title("ClimODE - ACC vs Lead Time")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("climode_eval_plot.png", dpi=150)
print("Saved climode_eval_plot.png")
