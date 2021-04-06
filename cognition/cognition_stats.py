#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 19:47:00 2020

@author: botond

Notes:
-this script performs linear regression on cognitive data from UKB.

"""

import os
import numpy as np
import pandas as pd
import itertools
import functools
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tqdm import tqdm
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match, check_assumptions
get_ipython().run_line_magic('cd', 'cognition')

# =============================================================================
# Setup
# =============================================================================

plt.style.use("ggplot")
sns.set_style("whitegrid")

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/cognition/"

# Inputs
CTRS = "diab"  # Contrast: diab or age
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
excl_sub = []  # SUbjects to exlude, if any
RLD = False  # Reload regressor matrices instead of computing them again

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
     "23324-2.0": "Processing_Speed"
     }

# Load data
data = pd \
    .read_csv(SRCDIR + "cognition/cognition_data.csv") \
    [["eid",
      *labels.keys()
     ]]

# Rename columns
data = data.rename(labels, axis=1)

# Exclude subjects
#data = data.query(f'eid not in {excl_sub}')

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
# Build regressor matrices
# =============================================================================

# Status
print(f"Building regressor matrices with contrast [{CTRS}].")

# Choose variables
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [diab, age, sex, college, ses, bmi, age_onset]
        ) \
        .drop("age_onset", axis=1)


# If contrast is age
if CTRS == "age":
    # Exclude subjects with T2DM
    regressors = regressors.query("diab == 0")

# Assign labels to features
features = [col for col in data.columns if "eid" not in col]

# Iterate over all features
for i, feat in enumerate(features):

    # Status
    print(f"Current feature: {feat}")

    # Clean feature
    # -----
    # Extract data, drop non-positive records
    data_feat = data.dropnaquery(f'`{feat}` > 0')[["eid", feat]]

    # Constrain ivs to only those samples that have y values
    regressors_y = regressors \
        .merge(data_feat, on="eid", how="inner") \
        .dropna(how="any") \

    # Temporarily remove y values
    regressors_clean = regressors_y.drop(feat, axis=1)

    # Check regressoriance of ivs
    # -----

    if CTRS == "age":

        # Interactions among independent variables
        var_dict = {
                "sex": "disc",
                "college": "disc",
                "ses": "disc",
                "bmi": "cont"
                }

        for name, type_ in var_dict.items():

            check_covariance(
                    regressors_clean,
                    var1=name,
                    var2="age",
                    type1=type_,
                    type2="cont",
                    save=True,
                    prefix=OUTDIR + f"covariance/pub_meta_cognition_covar_{feat}_"
                    )

            plt.close("all")

    if CTRS == "diab":

        # Interactions among independent variables
        var_dict = {
                "age": "cont",
                "sex": "disc",
                "college": "disc",
                "ses": "disc",
                "bmi": "cont"
                }

        for name, type_ in var_dict.items():

            check_covariance(
                    regressors_clean,
                    var1="diab",
                    var2=name,
                    type1="disc",
                    type2=type_,
                    save=True,
                    prefix=OUTDIR + f"covariance/pub_meta_cognition_covar_{feat}_"
                    )

            plt.close("all")

    # Perform matching
    # ------
    if (CTRS == "age") & (RLD == False):

        # Match
        regressors_matched = match(
                df=regressors_clean,
                main_var="sex",
                vars_to_match=["age"],
                N=1,
                random_state=1
                )

    if (CTRS == "diab") & (RLD == False):

        # Match
        regressors_matched = match(
                df=regressors_clean,
                main_var="diab",
                vars_to_match=["age", "sex"],
                N=1,
                random_state=1
                )

    # Save matched regressors matrix
    if RLD == False:
        regressors_matched.to_csv(OUTDIR + \
            f"regressors/pub_meta_cognition_matched_regressors_{feat}_{CTRS}.csv")

# %%
# =============================================================================
# Analysis
# =============================================================================

# Status
print(f"Fitting models for contrast [{CTRS}].")

# Dict to store stats that will be computed below
feat_stats = {}

# Iterate over all features
for i, feat in enumerate(features):

    # Status
    print(f"Current feature: {feat}")

    # Clean feature
    # -----
    # Extract data, drop non-positive records
    data_feat = data.query(f'`{feat}` > 0')[["eid", feat]]

    # Load regressors
    regressors_matched = pd.read_csv(OUTDIR + \
            f"regressors/pub_meta_cognition_matched_regressors_{feat}_{CTRS}.csv",
            index_col=0)

    # Reunite regressors and y values of the current feature
    sdf = regressors_matched.merge(data_feat, on="eid")

    # Get sample sizes
    sample_sizes = sdf.groupby("diab")["eid"].count()

    # Regression
    # -------
    # Formula
    if CTRS == "age":
        formula = f"{feat} ~ age + C(sex) + C(college) + C(ses) + bmi"
    if CTRS == "diab":
        formula = f"{feat} ~ C(diab) + age + C(sex) + C(college) + C(ses) + bmi"

    # Fit model
    model = smf.ols(formula, data=sdf)
    results = model.fit()

    # Monitor
    # ------
    # Save detailed stats report
    with open(OUTDIR + f"stats_misc/pub_meta_cognition_regression_table_{feat}" \
              f"_{CTRS}.html", "w") as f:
        f.write(results.summary().as_html())

    # Check assumptions
    check_assumptions(
            results,
            sdf,
            prefix=OUTDIR + \
            f"stats_misc/pub_meta_cognition_stats_assumptions_{feat}_{CTRS}"
            )

    # Plot across age
    if CTRS == "diab":
        plt.figure()
        plt.title(feat)
        sns.lineplot(data=sdf[[feat, "age", "diab"]], x="age", y=feat, hue="diab",
                     palette=sns.color_palette(["black", "red"]))
        plt.tight_layout()
        plt.savefig(OUTDIR + f"stats_misc/pub_meta_cognition_age-diab-plot_{feat}.pdf")
        plt.close()

    # Plot distribution of feature
    plt.figure()
    plt.title(feat)
    plt.hist(sdf[feat])
    plt.xlabel("score")
    plt.ylabel("count")
    plt.savefig(OUTDIR + f"stats_misc/pub_meta_cognition_dist_{feat}_{CTRS}.pdf")
    plt.tight_layout()
    plt.close()

    # Save results
    # -------
    # Normalization factor
    norm_fact = sdf.query('diab==0')[feat].mean()/100

    # Get relevant key for regressor
    rel_key = [key for key in results.conf_int().index.to_list() \
           if CTRS in key][0]

    # Get effect size
    tval = results.tvalues.loc[rel_key]
    beta = results.params.loc[rel_key]/norm_fact

    # Get confidence intervals
    conf_int = results.conf_int().loc[rel_key, :]/norm_fact

    # Get p value
    pval = results.pvalues.loc[rel_key]

    # Save stats as dict
    feat_stats[f"{feat}"] = [list(sample_sizes), tval,
                              pval, beta, np.array(conf_int)]


# Convert stats to df and correct p values for multicomp
df_out = pd.DataFrame.from_dict(
        feat_stats, orient="index",
        columns=["sample_sizes", "tval", "pval", "beta", "conf_int"]) \
        .reset_index() \
        .rename({"index": "label"}, axis=1) \
        .assign(**{"pval": lambda df: pg.multicomp(list(df["pval"]),
                                                   method="bonf")[1]})

# Flip tasks where higher score is worse
flip_tasks = ["Executive_Function", "Reaction_Time"]

# Temporarily assign index to df index column so I can slice into it using task names
df_out = df_out.set_index("label")

# Slice into df and multiply respective tasks with -1
df_out.loc[flip_tasks, ["tval", "beta", "conf_int"]] = \
    df_out.loc[flip_tasks, ["tval", "beta", "conf_int"]]*-1

# Undo to original indexing
df_out = df_out.reset_index()

# Save outputs
df_out.to_csv(OUTDIR + f"stats/pub_meta_cognition_stats_{CTRS}.csv")
