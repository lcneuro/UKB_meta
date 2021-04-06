#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 23:44:08 2021

@author: botond
"""

import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtc
import pingouin as pg
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib as mpl
from IPython import get_ipython
from tqdm import tqdm

get_ipython().run_line_magic('cd', '..')
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'volume')
get_ipython().run_line_magic('matplotlib', 'inline')

# =============================================================================
# Setup
# =============================================================================
# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "results/volume/stats/"
OUTDIR = HOMEDIR + "results/volume/"

#raise

# %%
# Settings
#contrast="t2dm"

# Open data
t2 = pd.read_csv(SRCDIR + "pub_meta_volume_stats_diab_46.csv", index_col=0) #.iloc[-10:, :].reset_index()
ag = pd.read_csv(SRCDIR + "pub_meta_volume_stats_age_46.csv", index_col=0) #.iloc[-10:, :].reset_index()

# Nuemrify interval data (from str to list of floats)
numerify_cols = lambda df: df.assign(
        **{
                "conf_int": lambda df: df["conf_int"].apply(
    lambda item: [float(val) for val in item[1:-1].split(" ") if val not in [" ", ""]]),
               "sample_sizes": lambda df: df["sample_sizes"].apply(
    lambda item: [int(val) for val in item[1:-1].split(", ") if val not in [" ", ""]])
    })

t2 = numerify_cols(t2)
ag = numerify_cols(ag)

# Add contraast columns. DO not edit these. They are used downstream!
t2["contrast"] = "t2dm"
ag["contrast"] = "age"

# Normalize age effects to t2dm
ag["beta"] = ag["beta"]*1

# Merge
df = pd.concat([ag, t2])

# Make a column with unique index
df["index"] = np.arange(len(df))

# Reorder per value of t2dm
order = df \
    .query('contrast == "age"') \
    .sort_values(by="beta")["index"] \
    .pipe(lambda df:
        pd.concat([df, df+df.shape[0]], axis=0)) \
    .to_frame() \
    .pipe(lambda df:
        df.assign(**{
                "ref": np.arange(df.shape[0])
                }))

df = df.merge(order, on="index", how="inner").sort_values(by="ref")

# For later use
sub_dfs = [df.query('contrast == "age"').reset_index(drop=True),
           df.query('contrast == "t2dm"').reset_index(drop=True)]

# Or just pick jsut 1 contrast at a time
#df = df.query(f'contrast == "{contrast}"')

# =============================================================================
# Plot
# =============================================================================

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

# Colors
colors_t2 = colors_from_values(
        np.array(list(-t2["beta"]) + [t2["beta"].min(), t2["beta"].max()]),
        "seismic_r")[:-2]

colors_age = colors_from_values(
        np.array(list(-ag["beta"]) + [ag["beta"].min() + 0, ag["beta"].max()]),
        "PuOr")[:-2]

colors = np.concatenate((colors_age, colors_t2), axis=0)

colors_dict = {i: colors[i] for i in range(len(colors))}

# Format labels
df["label"] = df["label"].str.replace("_", " ")

# Plotting
f = sns.FacetGrid(data=df, col="contrast", height=16, aspect=0.6,
                  sharex=False, despine=False) \
    .map_dataframe(sns.barplot, y="label", x="beta",
                   palette=colors_dict, hue="index", dodge=False,
                   linewidth=lw, edgecolor="black",
                   zorder=3, orient="h")

# Formatting
# =======

# For both axes
# ------

# Axis titles
ss = [t2["sample_sizes"][0], ag["sample_sizes"][0]]
title_texts = [
        f"Age (T2DM– only, sex-matched)\nN$_{{}}$={ss[1][0]:,}",
        f"T2DM (T2DM+ vs. T2DM–, age and sex-Matched)\nN$_{{T2DM+}}$={ss[0][1]:,}, " \
        f"N$_{{T2DM–}}$={ss[0][0]:,}"
        ]


# Loop through
for i, ax in enumerate(f.axes[0]):

    # Unpack
    sub_df = sub_dfs[i]

    # Remove title
    ax.set_title(title_texts[i])

    # Add grid
    ax.grid(zorder=0)

    # Add x=0 axvline
    ax.axvline(x=0, linewidth=3, color="black")

    # Add in errobars
    for _, item in tqdm(enumerate(sub_df.iterrows()), total=len(sub_df)):

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


# Axis specific formatting
# ========

# Age
# -----
ax = f.axes[0][0]
ax.set_xlim([-1.19, 0.59])
ax.spines['right'].set_visible(False)
ax.set_xlabel("Percentage change in gray matter volume\nacross age (% per year)")

# T2DM
# ----
ax = f.axes[0][1]
ax.set_xlim([-7.9, 5.3])
ax.set_xlabel("Percentage difference in gray matter volume\nT2DM+ vs. T2DM– (%)")

# Figure formatting
# ------

# Add common suptitle
plt.suptitle("\t\t\t"*2 + "Gray matter volume changes associated with age and T2DM: " \
             "UK Biobank dataset")

## Add common x label
#plt.gcf().text(0.6, 0.03, "Change In Gray Matter Volume (%)", ha='center',
#        fontsize=14*fs, fontweight="bold")


plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_volume.pdf",
            transparent=True)
plt.close("all")

#from scipy import stats
#stats.pearsonr(ag["beta"], t2["beta"])
#plt.scatter(ag["beta"], t2["beta"])
#plt.xlim([-0.75, 0])
#plt.ylim([-3, 0])