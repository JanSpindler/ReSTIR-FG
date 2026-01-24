import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use a clean style
sns.set(style="whitegrid")

# Scene names
scenes = [
    "BistroCookie", "BistroTable", "CausticGlass",
    "SibenikStatue", "VeachAjarFar", "VeachAjarNear"
]

# Base values and increments for ReSTIR FG
restir_fg_base = [34, 31, 19.8, 31.9, 21.1, 22.3]
first_hit = [70 - 34, 69 - 31, 22.9 - 19.8, 39.4 - 31.9, 25.7 - 21.1, 27.6 - 22.3]
adaptive_light = [100 - 70, 116 - 69, 23.6 - 22.9, 39.4 - 39.4, 26.8 - 25.7, 27.8 - 27.6]
prefix_restir = [118 - 100, 118 - 116, 26.9 - 23.6, 43 - 39.4, 28.5 - 26.8, 30.5 - 27.8]
indirect_caustic = [120 - 118, 120 - 118, 27.9 - 26.9, 46.8 - 43, 30.4 - 28.5, 32.7 - 30.5]

# ReSTIR PT values
restir_pt = [112, 127, 61, 33.3, 44.8, 62.8]

# Bar width and positions
bar_width = 0.35
x = np.arange(len(scenes))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Stacked bars for ReSTIR FG
ax.bar(x - bar_width/2, restir_fg_base, bar_width, label='ReSTIR FG Base')
ax.bar(x - bar_width/2, first_hit, bar_width, bottom=restir_fg_base, label='First Hit Guiding')
ax.bar(x - bar_width/2, adaptive_light, bar_width,
       bottom=np.array(restir_fg_base) + np.array(first_hit),
       label='Adaptive Light Sampler')
ax.bar(x - bar_width/2, prefix_restir, bar_width,
       bottom=np.array(restir_fg_base) + np.array(first_hit) + np.array(adaptive_light),
       label='Prefix ReSTIR')
ax.bar(x - bar_width/2, indirect_caustic, bar_width,
       bottom=np.array(restir_fg_base) + np.array(first_hit) + np.array(adaptive_light) + np.array(prefix_restir),
       label='Indirect Caustic Guiding')

# Separate bars for ReSTIR PT
ax.bar(x + bar_width/2, restir_pt, bar_width, label='ReSTIR PT')

# Labels and title
ax.set_xlabel('Scenes', fontsize=18)
ax.set_ylabel('Frametime [ms]', fontsize=18)
ax.set_title('Performance Comparison: ReSTIR FG (stacked) vs ReSTIR PT', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(scenes, rotation=45, fontsize=16)
ax.legend(fontsize=14)

plt.tight_layout()
plt.show()
