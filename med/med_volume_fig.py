#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:40:49 2021

@author: botond
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'med')
get_ipython().run_line_magic('matplotlib', 'inline')


# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/med/volume/"

# Inputs
CTRS = "metfonly_unmed"  # Contrast: diab or age
PARC = 46  # Type of parcellation to use, options: 46 or 139

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

# Load df
df = pd \
    .read_csv(OUTDIR + f"stats/pub_meta_med_volume_stats_{CTRS}_{PARC}.csv",
              index_col=0) \
    .pipe(lambda df:
        df.assign(**{
                "label": df["label"].str.replace("_", " "),
                "index": np.arange(len(df)),
                "conf_int": lambda df: df["conf_int"].apply(
                        lambda item: [float(val) for val in item[1:-1].split(" ") \
                                      if val not in [" ", ""]]),
                "sample_sizes": lambda df: df["sample_sizes"].apply(
                        lambda item: [int(val) for val in item[1:-1].split(", ") \
                                      if val not in [" ", ""]])
                }))

# Colors
colors = colors_from_values(
        np.array(list(-df["beta"]) + [df["beta"].min(), df["beta"].max()]),
        "coolwarm_r")[:-2]

colors_dict = {i: colors[i] for i in range(len(colors))}

# Make figure
plt.figure(figsize=(19.2, 22))

# Plot
sns.barplot(data=df, y="label", x="beta",
            palette=colors_dict, hue="index", dodge=False,
            linewidth=lw, edgecolor="black",
            zorder=3, orient="h")

# Turn legend off
plt.legend([],[], frameon=False)

# Get current axis
ax = plt.gca()

# Add grid
ax.grid(zorder=0)

# Add x=0 axvline
ax.axvline(x=0, linewidth=3, color="black")

# Add in errobars
for _, item in tqdm(enumerate(df.iterrows()), total=len(df)):

    y = item[0]
    ss, p, x, conf_int  = item[1][["sample_sizes", "pval", "beta", "conf_int"]]

    conf_dist = abs(x - np.array(conf_int))[:, None]

    ax.errorbar(x, y, xerr=conf_dist, capsize=3, capthick=lw,
                 elinewidth=lw, color="black", zorder=100)

#    text = pformat(p) + p2star(p ) if PRINT_P \
#            else p2star(p)
#    ha = "left"
#    x = 3.2

    text = p2star(p)

    ha = "right" if x < 0 else "left"

    xoffset = 0.15 if x == 0 else 0.02
    xtext = min(conf_int) + xoffset if x < 0 else max(conf_int) - xoffset
    ytext = y + 0.2

    ax.annotate("   " + text + "   ", xy=[xtext, ytext],
                 ha=ha, va="center", fontsize=12*fs, fontweight="bold")


# Add spines
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(2)
    ax.spines[sp].set_color("black")

# Add labels
ss = df["sample_sizes"]
ax.set_xlabel(f"Percentage difference in gray matter volume\n{CTRS} (% of avg)")

plt.title("Gray matter volume as a function of metformin medication status:" \
          f"\nUK Biobank dataset ({CTRS}, age, sex and disease duration matched)\n" \
          + f"(N$_{{metf+}}$={ss[0][1]:,}, N$_{{metf-}}$={ss[0][0]:,})")
ax.set_ylabel("")

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig(OUTDIR + f"figures/JAMA_meta_figure_med_volume_{CTRS}.pdf",
            transparent=True)
plt.close("all")