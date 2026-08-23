import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = Path('IVSF/outputs/adaptive_badnet_ivsf/ivsf_adaptive_badnet_plot_data.csv')
out_dir = csv_path.parent
out_dir.mkdir(parents=True, exist_ok=True)

x = []
y = []
with csv_path.open('r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        x.append(int(row['leaked_samples']))
        y.append(float(row['validation_with_leakage']))

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 16,
})

fig, ax = plt.subplots(figsize=(8.0, 5.2))
ax.plot(
    x,
    y,
    linestyle='--',
    linewidth=2.0,
    color='blue',
    marker='s',
    markersize=6,
    label='Validation w. Leakage',
)

ax.set_xlim(0, 5000)
ax.set_ylim(0, 100)
ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

ax.set_xlabel('# of leaked samples')
ax.set_ylabel('Detection rate (%)')
ax.grid(True, linestyle='--', color='gray', alpha=0.4)
ax.legend(loc='upper right', frameon=True)

ax.set_title('(a) Adaptive backdoor attacks')

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

fig.tight_layout()
fig.savefig(out_dir / 'ivsf_adaptive_badnet_result.png', dpi=300)
print(f'Saved figure: {out_dir / "ivsf_adaptive_badnet_result.png"}')
print(f'Saved csv: {csv_path}')
