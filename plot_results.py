import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("eval_results.json") as f:
    results = json.load(f)

lead_times = sorted(results.keys(), key=lambda x: int(x.replace("h", "")))
hours = [int(lt.replace("h", "")) for lt in lead_times]
rmses = [results[lt]["t2m"]["rmse_mean"] for lt in lead_times]
accs = [results[lt]["t2m"]["acc_mean"] for lt in lead_times]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(hours, rmses, "o-", linewidth=2, markersize=6, color="tab:blue")
axes[0].set_xlabel("Lead Time (hours)")
axes[0].set_ylabel("Lat-weighted RMSE (K)")
axes[0].set_title("ClimODE - 2m Temperature RMSE")
axes[0].grid(True, alpha=0.3)

axes[1].plot(hours, accs, "o-", linewidth=2, markersize=6, color="tab:orange")
axes[1].set_xlabel("Lead Time (hours)")
axes[1].set_ylabel("ACC")
axes[1].set_title("ClimODE - 2m Temperature ACC")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("climode_eval_plot.png", dpi=150)
print("Saved climode_eval_plot.png")
