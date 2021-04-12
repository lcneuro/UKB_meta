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
get_ipython().run_line_magic('cd', 'cognition')

# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/cognition/"

# Inputs
CTRS = "duration"  # Contrast: diab or age
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
DUR_CO = 10  # Year to separate subjects along duration <x, >=x
PARC = 46  # Type of parcellation to use, options: 46 or 139
RLD = 1 # Reload regressor matrices instead of computing them again

print("\nRELOADING REGRESSORS!\n") if RLD else ...

#raise

# %%
# =============================================================================
# Load data
# =============================================================================

# Load cognitive data
# -------
# Labels
labels = {
     "4282-2.0": "Short_Term_Memory",
     "6350-2.0": "Executive_Function",
     "20016-2.0": "Abstract_Reasoning",
     "20023-2.0": "Reaction_Time",
     "23324-2.0": "Processing_Speed",
     }

# Load data
data = pd \
    .read_csv(SRCDIR + "cognition/cognition_data.csv") \
    [["eid",
      *labels.keys()
     ]]

# Rename columns
data = data.rename(labels, axis=1)

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

# Get regressor columns for later
reg_cols = regressors.columns

# Merge domain data and clean out invalied entries
data_merged = data[data>0].dropna()

# Get data columns for later
data_cols = data_merged.columns

# Merge regressors with data
regressors_y = regressors.merge(data_merged, on="eid", how="inner")

# Drop data columns
regressors_clean = regressors_y.drop(data_cols[1:], axis=1)

# Save full regressor matrix
regressors_clean.to_csv(OUTDIR + f"regressors/pub_meta_cognition_acceleration" \
                        f"_full_regressors_{CTRS}.csv")

# Interactions among independent variables
var_dict = {
        "age": "cont",
        "sex": "disc",
        "college": "disc",
        }

for name, type_ in var_dict.items():

    check_covariance(
            regressors_clean.query(f'{CTRS} == {CTRS}'),
            var1=CTRS,
            var2=name,
            type1="cont",
            type2=type_,
            save=True,
            prefix=OUTDIR + "covariance/pub_meta_cognition_acceleration_covar"
            )

    plt.close("all")

if RLD == False:
    # Match
    regressors_matched = match_multi(
            df=regressors_clean,
            main_var="duration_group",
            vars_to_match=["age", "sex", "college"],
            N=1,
            random_state=1
            )

if RLD == False:
    # Save matched regressors matrix
    regressors_matched \
        .reset_index(drop=True) \
        .to_csv(OUTDIR + f"regressors/pub_meta_cognition_acceleration" \
                f"_matched_regressors_{CTRS}.csv")


# %%
# =============================================================================
# Statistics
# =============================================================================

# Prep
# ------
# Get regressors
regressors_matched = pd.read_csv(
        OUTDIR + f"regressors/pub_meta_cognition_acceleration_matched_regressors_{CTRS}.csv",
        index_col=0)

# Merge cognitive data with regressors, standardize, and compress into an aggregated measure
df = regressors_matched \
    .merge(data_merged, on="eid") \
    .set_index(list(reg_cols)) \
    .pipe(lambda df: df.assign(**{
        "Short_Term_Memory": df["Short_Term_Memory"],
         "Executive_Function": -1*df["Executive_Function"],
         "Abstract_Reasoning": df["Abstract_Reasoning"],
         "Reaction_Time": -1*df["Reaction_Time"],
         "Processing_Speed": df["Processing_Speed"]
         })) \
    .pipe(lambda df:
        ((df - df.mean(axis=0))/df.std(axis=0)).mean(axis=1)) \
    .rename("score") \
    .reset_index() \

# Take nondiabetic subs
sdf = df.query('diab == 1')

# Fit
# ------
# Feature
feat = "score"

# Fit the model to get brain age
model = smf.ols(f"{feat} ~ age + C(sex) + C(college) + duration", data=sdf)
results = model.fit()

# Monitor
# --------

#print(results.summary())

# Save detailed stats report
with open(OUTDIR + f"stats_misc/pub_meta_cognition_acceleration_regression_table_{feat}" \
          f"_{CTRS}.html", "w") as f:
    f.write(results.summary().as_html())

## Check assumptions
#check_assumptions(
#        results,
#        sdf,
#        prefix=OUTDIR + \
#        f"stats_misc/pub_meta_cognition_acceleration_stats_assumptions_{feat}_{CTRS}"
#        )


# Results
# -------

# Calculate acceleration of aging in additional year/year
acc_res = results.params["duration"]/results.params["age"]

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
palette = sns.color_palette(["black", "tomato", "darkred"])

# Content
# -----

# Make figure
plt.figure(figsize=(4.25, 5.5))

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
       f"$\mathbf{{H_0}}$:  $\mathrm{{\\beta}}$$\mathbf{{_t}}$ = 0\n" \
       f"$\mathbf{{H_1}}$:  $\mathrm{{\\beta}}$$\mathbf{{_t}}$ ≠ 0\n" \
       f"T = {tval:.1f}\n{pformat(pval)}" \
       f"{p2star(pval)}"

plt.annotate(text, xycoords="axes fraction", xy=[0.279, 0.03],
             fontsize=8*fs, fontweight="bold", ha="center")


# Format
# ----

# Title
ttl = plt.title("Cognitive performance across age and T2DM disease duration:\n" \
          f"UK Biobank dataset\n"
          f"N$_{{≥10 years}}$={int(gdf.shape[0]/3)}, " \
          f"N$_{{0-9 years}}$={int(gdf.shape[0]/3)}, " \
          f"N$_{{HC}}$={int(gdf.shape[0]/3)}\n"
          )
ttl.set_x(ttl.get_position()[0]-0.056)

plt.xlabel("Age group (year)")
#plt.ylabel("Gray matter cognition delineated\nbrain age (y)")

plt.ylabel("Standardized cognitive performance score")

legend_handles, _ = plt.gca().get_legend_handles_labels()
[ha.set_linewidth(5) for ha in legend_handles]

plt.legend(title="T2DM disease duration",
           handles=legend_handles[::-1],
           labels=["≥10 years", "0-9 years", "HC"],
           loc=1)

plt.gca().xaxis.tick_bottom()
plt.gca().yaxis.tick_left()

for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(0.5*lw)
    plt.gca().spines[sp].set_color("black")

plt.gca().xaxis.grid(False)
plt.tight_layout()

# Save
# ----

plt.tight_layout(rect=[0, 0., 1, 0.99])
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_cognition_acceleration.pdf",
            transparent=True)
plt.close("all")

