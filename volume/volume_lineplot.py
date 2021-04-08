#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:58:20 2021

@author: botond
"""

import os
import numpy as np
import pandas as pd
import itertools
import functools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtc
import seaborn as sns
import pingouin as pg
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tqdm import tqdm
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match_multi, check_assumptions
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'volume')

# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/volume/"

# Inputs
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.

#raise

# %%
# =============================================================================
# Load data
# =============================================================================

# Load volume data
# -------
# Load atrophy data
data = pd.read_csv(SRCDIR + "volume/volume_data.csv").drop(["age", "gender"], axis=1) \
    [["eid", '25005-2.0']].rename({'25005-2.0': "volume"}, axis=1)

# Load regressors
# ------
# Age
age = pd.read_csv(SRCDIR + "ivs/age.csv", index_col=0)[["eid", "age-2"]] \
    .rename({"age-2": "age"}, axis=1)

# Sex
sex = pd \
    .read_csv(SRCDIR + "ivs/sex.csv", index_col=0)[["eid", "sex"]] \
    .set_axis(["eid", "sex"], axis=1)

# Diabetes diagnosis
diab = pd.read_csv(SRCDIR + "ivs/diab.csv", index_col=0)[["eid", "diab-2"]] \
    .rename({"diab-2": "diab"}, axis=1) \
    .query('diab >= 0')

# College
college = pd.read_csv(SRCDIR + "ivs/college.csv", index_col=0)[["eid", "college"]] \

# Ses
ses = pd.read_csv(SRCDIR + "ivs/ses.csv", index_col=0)[["eid", "ses"]]

# BMI
bmi = pd.read_csv(SRCDIR + "ivs/bmi.csv", index_col=0)[["eid", "bmi-2"]] \
    .rename({"bmi-2": "bmi"}, axis=1) \
    .dropna(how="any")

# Age of diabetes diagnosis (rough estimate!, averaged)
age_onset = pd \
    .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
    .set_index("eid") \
    .mean(axis=1) \
    .rename("age_onset") \
    .reset_index()

# Remove diabetic subjects with missing age of onset OR have too early age of onset
# which would suggest T1DM. If age of onset is below T1DM_CO, subject is excluded.
age_onset = age_onset \
    .merge(diab, on="eid", how="inner") \
    .query(f'(diab==0 & age_onset!=age_onset) or (diab==1 & age_onset>={T1DM_CO})') \
    [["eid", "age_onset"]]

# %%
# =============================================================================
# Transform
# =============================================================================

# Status
print(f"Transforming.")


# Choose variables and group per duration
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [age, sex, college, ses, bmi, diab, age_onset]
        ) \
        .drop("age_onset", axis=1) \

# Merge and standardize
df = regressors \
    .merge(data, on="eid") \
    .pipe(lambda df: df.assign(**{"age_group":
            pd.cut(df["age"], bins=np.arange(0, 100, 5)).astype(str)
            })) \
    .sort_values(by="age") \
    .query('age_group not in ["(40, 45]", "(45, 50]", "(75, 80]"]')

# %%
# =============================================================================
# Statistics
# =============================================================================

# Prep
# ------
# matching needed? YES! + stats needed for both
## Fit
## ------
#
## Fit the model to get brain age
#model = smf.ols(f"volume ~ age + C(diab) + C(sex) + C(college) + C(ses) + bmi", data=df)
#results = model.fit()
#
## Monitor
## --------
#
## Save detailed stats report
#with open(OUTDIR + f"stats_misc/pub_meta_volume_regression_table_{feat}" \
#          f"_{CTRS}.html", "w") as f:
#    f.write(results.summary().as_html())
#
## Check assumptions
#check_assumptions(
#        results,
#        sdf,
#        prefix=OUTDIR + \
#        f"stats_misc/pub_meta_volume_stats_assumptions_{CTRS}"
#        )

# %%
# =============================================================================
# Plot
# =============================================================================

# Status
print(f"Plotting.")

# Prep
# -----
# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs


# Sample sizes
#print("Sampe sizes, age info:\n", gdf.groupby(['duration_group', 'age_group'])["age"].describe())

# Colors
palette = sns.color_palette(["black", "red"])

# Content
# -----

# Make figure
plt.figure(figsize=(4.25, 5.5))

# Create plot
sns.lineplot(data=df, x="age_group", y="volume",
         hue="diab", ci=68, err_style="bars",
         marker="o", linewidth=1*lw, markersize=2*lw, err_kws={"capsize": 2*lw,
                                                         "capthick": 1*lw,
                                                         "elinewidth": 1*lw},
         sort=False, palette=palette)

# Annotate stats
#tval, pval = results.tvalues["duration"], results.pvalues["duration"]
#text = f"Time since T2DM diagnosis\nas a continuous linear factor:\n" \
#       f"$\mathbf{{H_0}}$:  $\mathrm{{\\beta}}$$\mathbf{{_t}}$ = 0\n" \
#       f"$\mathbf{{H_1}}$:  $\mathrm{{\\beta}}$$\mathbf{{_t}}$ ≠ 0\n" \
#       f"T = {tval:.1f}\n{pformat(pval)}" \
#       f"{p2star(pval)}"

#plt.annotate(text, xycoords="axes fraction", xy=[0.85, 0.6],
#             fontsize=14*fs, fontweight="bold", ha="center")


# Format
# ----

# Title
plt.title("Gray matter atrophy across age and T2DM status:\n" \
          f"UK Biobank dataset ")

plt.xlabel("Age group (year)")
#plt.ylabel("Gray matter volume delineated\nbrain age (y)")

plt.ylabel("Gray matter volume (voxel count)")
plt.gca().yaxis.set_major_formatter(mtc.FuncFormatter
       (lambda x, pos: f"{x/1e5:.1f}"))
plt.annotate("×10$^5$", xy=[0, 1.03], xycoords="axes fraction",
             fontsize=8*fs, va="center")

legend_handles, _ = plt.gca().get_legend_handles_labels()
[ha.set_linewidth(5) for ha in legend_handles]

plt.legend(handles=legend_handles[::-1],
           labels=["T2DM+", "T2DM-"],
           loc=1)

plt.gca().xaxis.tick_bottom()
plt.gca().yaxis.tick_left()

for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(2)
    plt.gca().spines[sp].set_color("black")

plt.gca().yaxis.grid(True)
plt.tight_layout()

# Save
# ----

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_volume_lineplot.pdf",
            transparent=True)
plt.close("all")

