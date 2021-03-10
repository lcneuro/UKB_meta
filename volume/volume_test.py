#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 19:47:00 2020

@author: botond

Notes:
-This script is for graphing sets of features for the keck report.
-This version in particular is for cognitive features. There is a another,
very similar script for atrophy.
-I normalize with head size, but only when merging regions, not at the
beginning!

# Caption
# TODO: wrong sample size used in duration!

"""

import os
import numpy as np
import pandas as pd
import itertools
import functools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtc
import pingouin as pg
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib as mpl
from scipy import stats

from helpers import regression_helpers

# =============================================================================
# Setup
# =============================================================================

plt.style.use("ggplot")

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/volume/"

# Inputs
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
EFFS = "beta"
PRINT_P = True  # Print p values on plot or not

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

# Load meta data
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
    .query(f'(diab==0) or (diab==1 & age_onset>={T1DM_CO})') \
    [["eid", "age_onset"]]

# =============================================================================
# Test linear models
# =============================================================================

# Choose variables
meta = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid"),
        [diab, age, sex, college, ses, bmi, age_onset]
        )

# Build sdf
y = data['Volume of grey matter (normalised for head size)'].rename("feat").reset_index()
sdf = y.merge(meta, on="eid", how="inner")

# Fit model
model = smf.ols(f"feat ~ C(diab) + age + C(sex) + C(college) + C(ses) + bmi", data=sdf)
results = model.fit()

# Print results
results.summary()

# Check assumptions
regression_helpers.check_assumptions(results, sdf)

# Interactions among independent variables
regression_helpers.check_covariance(meta, var1= "diab", var2="bmi", type1="disc", type2="cont")
regression_helpers.check_covariance(meta, var1= "diab", var2="age", type1="disc", type2="cont")
regression_helpers.check_covariance(meta, var1= "diab", var2="sex", type1="disc", type2="disc")
regression_helpers.check_covariance(meta, var1= "diab", var2="college", type1="disc", type2="disc")
regression_helpers.check_covariance(meta, var1= "diab", var2="ses", type1="disc", type2="disc")

# %%
# Matching
# ------

# Variable to match for
main_var = "diab"

# Number of ctrl samples to take for every single exp item
N = 1

# Variable(s) to match for
vars_to_match = ["age", "sex"]

# Separate items per main variable
exp_subs = sdf.query(f"{main_var} == 1")
ctrl_subs = sdf.query(f"{main_var} == 0")

# List of matched items, later to serve as a df
mdf_list = []

# List to store number of available subject
candidate_numbers_list = []

# Iterate over all subjects positive to the treatment
for i, exp_sub in enumerate(exp_subs.iterrows()):

    # Find control subjects that match along variables
    query_statement = " & ".join([f'{var} == {exp_sub[1][var]}' \
                                  for var in vars_to_match])

    candidates = ctrl_subs.query(query_statement)

    # Store numbers
    candidate_numbers_list.append(len(candidates))

    # If there is at least 1 match
    if candidates.shape[0] > 0:

        # Pick from candidates randomly
        picked_ctrl_subs = candidates.sample(N)

        # If found: Take out subject from ctrl_subs hat
        ctrl_subs = ctrl_subs \
            .merge(
                picked_ctrl_subs,
                on=ctrl_subs.columns.to_list(),
                how="left",
                indicator=True
                ) \
            .query('_merge != "both"').drop("_merge", axis=1)

        # If found: add both subjects to mdf
        mdf_list.append(exp_sub[1].to_frame().T)
        mdf_list.append(picked_ctrl_subs)

    else:
        print(f"No match found for: {exp_sub}")

# Concat into df
mdf = pd.concat(mdf_list, axis=0)

# Analyze candidate availability
candidate_numbers = pd.DataFrame(candidate_numbers_list, columns=["count"])
print("Matching info:\n", candidate_numbers.describe())

# %%
# Fit model
model = smf.ols(f"feat ~ C(diab) + age + C(sex) + C(college) + C(ses) + bmi", data=mdf)
results = model.fit()

# Print results
results.summary()

# %%
#TODO: figure out the order of things
"""
Address covariance outside
Print these things into pdf
Perform matching outside
Fit model
Check assumptions
Save fit, plot, reports into a separate file

"""
#TODO: write up for all regions
#TODO: make it adaptable to 139 regions as well
#TODO: print outputs from assumptions and interaction in all cases

#TODO: maybe: do ratio matching
#TODO: maybe: stratify across age
#TODO: reorganize plot
