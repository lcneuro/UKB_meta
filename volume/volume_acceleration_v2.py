#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 11:13:40 2021

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
CTRS = "duration"  # Contrast: diab or age
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
DUR_CO = 10  # Year to separate subjects along duration <x, >=x
PARC = 46  # Type of parcellation to use, options: 46 or 139
RLD = 1 # Reload regressor matrices instead of computing them again

print("\nRELOADING REGRESSORS!\n") if RLD else ...

raise

# %%
# =============================================================================
# Load data
# =============================================================================

# Load volume data
# -------
# Load atrophy data
data = pd.read_csv(SRCDIR + "volume/volume_data.csv").drop(["age", "gender"], axis=1)

# Load labels
labels = pd \
    .read_csv(SRCDIR + "volume/volume_labels.csv",
                     index_col=0, header=0, names=["ID", "label"]) \
    .set_index("ID").to_dict()["label"]

# Load head size normalization factor
head_norm = pd.read_csv(SRCDIR + "volume/head_size_norm_fact.csv")
head_norm = data.merge(head_norm, on="eid", how="inner")[["eid", "norm_fact"]]

# Rename columns
data = data.rename(labels, axis=1).set_index("eid")

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

# Age of diabetes diagnosis (rough estimate!, averaged)
age_onset = pd \
    .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
    .set_index("eid") \
    .mean(axis=1) \
    .rename("age_onset") \
    .reset_index()

# Duration
duration = age_onset \
    .merge(age, on="eid", how="inner") \
    .pipe(lambda df: df.assign(**{
            "duration": df["age"] - df["age_onset"]})) \
    .dropna(how="any") \
    [["eid", "duration"]]

# Remove diabetic subjects with missing age of onset OR have too early age of onset
# which would suggest T1DM. If age of onset is below T1DM_CO, subject is excluded.
age_onset = age_onset \
    .merge(diab, on="eid", how="inner") \
    .query(f'(diab==0 & age_onset!=age_onset) or (diab==1 & age_onset>={T1DM_CO})') \
    [["eid", "age_onset"]]

# %%
# =============================================================================
# Build regressor matrix
# =============================================================================

# Status
print(f"Building regressor matrix with contrast [{CTRS}]")


# Choose variables and group per duration
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [age, sex, college, diab, age_onset]
        ) \
        .drop("age_onset", axis=1) \
        .merge(duration, on="eid", how="left") \
        .pipe(lambda df: df.assign(**{
        "duration_group":
            df["duration"] \
                .apply(lambda x: \
                       "ctrl" if x!=x else \
                       f"<{DUR_CO}" if x<DUR_CO else \
                       f">={DUR_CO}" if x>=DUR_CO else np.nan)
                }
            ))

# Inner merge regressors with a gm column to make sure all included subjects have data
y = data['Volume of grey matter (normalised for head size)'].rename("feat").reset_index()
regressors_y = y.merge(regressors, on="eid", how="inner")

# Drop feat column
regressors_clean = regressors_y.drop(["feat"], axis=1)

# Save full regressor matrix
regressors_clean.to_csv(OUTDIR + f"regressors/pub_meta_volume_acceleration_full_regressors_{CTRS}.csv")

# Interactions among independent variables
var_dict = {
        "age": "cont",
        "sex": "disc",
        "college": "disc",
        }

for name, type_ in var_dict.items():

    if RLD == False:

        check_covariance(
                regressors_clean.query(f'{CTRS} == {CTRS}'),
                var1=CTRS,
                var2=name,
                type1="cont",
                type2=type_,
                save=True,
                prefix=OUTDIR + "covariance/pub_meta_volume_acceleration_covar"
                )

        plt.close("all")


        # Match
        regressors_matched = match_multi(
                df=regressors_clean,
                main_var="duration_group",
                vars_to_match=["age", "sex", "college"],
                N=1,
                random_state=100
                )

if RLD == False:
    # Save matched regressors matrix
    regressors_matched \
        .reset_index(drop=True) \
        .to_csv(OUTDIR + f"regressors/pub_meta_volume_acceleration_matched_regressors_{CTRS}.csv")

# %%
# =============================================================================
# Statistics
# =============================================================================

# Prep
# ------
# Get regressors
regressors_matched = pd.read_csv(
        OUTDIR + f"regressors/pub_meta_volume_acceleration_matched_regressors_{CTRS}.csv",
        index_col=0)

# Join regressors with data
y = data['Volume of grey matter (normalised for head size)'].rename("Whole_Brain").reset_index()
df = regressors_matched.merge(y, on="eid", how="inner")

# Take nondiabetic subs
sdf = df.query('diab == 1')

# Fit
# ------
# Feature
feat = "Whole_Brain"

# Fit the model to get brain age
model = smf.ols(f"{feat} ~ age + C(sex) + C(college) + duration", data=sdf)
results = model.fit()

# Monitor
# --------

#print(results.summary())

# Save detailed stats report
with open(OUTDIR + f"stats_misc/pub_meta_volume_acceleration_regression_table_{feat}" \
          f"_{CTRS}.html", "w") as f:
    f.write(results.summary().as_html())

# Check assumptions
check_assumptions(
        results,
        sdf,
        prefix=OUTDIR + \
        f"stats_misc/pub_meta_volume_acceleration_stats_assumptions_{feat}_{CTRS}"
        )

# Results
# -------

# Calculate acceleration of aging in additional year/year
acc_res = results.params["duration"]/results.params["age"]

# Covariance matrix of coefficients
print(results.cov_params())

"""
# CI for the ratio is computed using an online tool (Fieller method):
https://www.graphpad.com/quickcalcs/errorProp1/?Format=SEM

An alternative approach would be to bootstrap using sigmas and covariances.
"""

print(f"Acceleration: +{acc_res:.2g} year/year", f'p={results.pvalues["duration"]:.2g},', \
      "significant" if results.pvalues["duration"]<0.05 else "not significant!")


# %%
# =============================================================================
# Visualize
# =============================================================================

# Prep
# -----
# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs
lw = lw*1.5

# Get data
gdf = df.copy()

# Make age groups
gdf = gdf \
    .pipe(lambda df:
        df.assign(**{"age_group": pd.cut(df["age"], np.arange(0, 100, 5),
               include_lowest=True, precision=0).astype(str)})) \
    .query('age_group not in ["(40, 45]", "(45, 50]", "(75, 80]"]')

# Sort
gdf = gdf.sort_values(by=["age", "duration"], na_position="first")

# Sample sizes
print("Sampe sizes, age info:\n", gdf.groupby(['duration_group', 'age_group'])["age"].describe())

# Colors
palette = sns.color_palette(["black", "darkorange", "red"])

# Content
# -----

# Make figure
plt.figure(figsize=(5, 4))

# Create plot
sns.lineplot(data=gdf, x="age_group", y=feat,
         hue="duration_group", ci=68, err_style="bars",
         marker="o", linewidth=1*lw, markersize=2*lw, err_kws={"capsize": 2*lw,
                                                         "capthick": 1*lw,
                                                         "elinewidth": 1*lw},
         sort=False, palette=palette)

# Annotate stats
tval, pval = results.tvalues["duration"], results.pvalues["duration"]
text = f"T2DM disease duration\nas a continuous linear factor:\n" \
       f"${{H_0}}$:  $\mathrm{{\\beta}}$${{_t}}$ = 0\n" \
       f"${{H_1}}$:  $\mathrm{{\\beta}}$${{_t}}$ ≠ 0\n" \
       f"T = {tval:.1f}; {pformat(pval)}{p2star(pval)}"

plt.annotate(text, xycoords="axes fraction", xy=[0.24, 0.03],
             fontsize=8*fs, fontweight="regular", ha="center")


# Format
# ----

# Title
ttl = plt.title("Gray matter atrophy across age and T2DM disease duration:\n" \
          f"UK Biobank dataset \n"
          f"    N$_{{≥10y}}$={int(gdf.shape[0]/3)}, " \
          f"N$_{{0–9y}}$={int(gdf.shape[0]/3)}, " \
          f"N$_{{HC}}$={int(gdf.shape[0]/3)}"
          )
ttl.set_x(ttl.get_position()[0]-0.056)
ttl.set_y(ttl.get_position()[1]+3)

plt.xlabel("Age group (year)")
#plt.ylabel("Gray matter volume delineated\nbrain age (y)")

plt.ylabel("Gray matter volume (voxel count)")
plt.gca().yaxis.set_major_formatter(mtc.FuncFormatter
       (lambda x, pos: f"{x/1e5:.1f}"))
plt.annotate("×10$^5$", xy=[0, 1.03], xycoords="axes fraction",
             fontsize=8*fs, va="center")

legend_handles, _ = plt.gca().get_legend_handles_labels()
[ha.set_linewidth(5) for ha in legend_handles]

plt.legend(title="T2DM disease duration",
           handles=legend_handles[::-1],
           labels=["≥10 years", "0–9 years", "HC"],
           loc=1)

plt.gca().xaxis.tick_bottom()
plt.gca().yaxis.tick_left()

for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(0.75*lw)
    plt.gca().spines[sp].set_color("black")

plt.gca().xaxis.grid(False)
plt.tight_layout()

# Save
# ----

plt.tight_layout(rect=[0, 0., 1, 1.02])
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_volume_acceleration.pdf",
            transparent=True)
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_volume_acceleration.svg",
            transparent=True)
plt.close("all")
