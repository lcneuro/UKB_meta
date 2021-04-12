#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 19:47:00 2020

@author: botond

Notes:
-this script performs linear regression on gray matter volume data
-I normalize with head size, but only when merging regions, not at the
beginning!


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
from helpers.regression_helpers import check_covariance, match_mah, match_cont, check_assumptions
get_ipython().run_line_magic('cd', 'volume')

# =============================================================================
# Setup
# =============================================================================

plt.style.use("ggplot")

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/volume/"

# Inputs
CTRS = "diab"  # Contrast: diab or age
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
PARC = 46  # Type of parcellation to use, options: 46 or 139
## to exlucde due to abnormal total gray matter volumes
excl_region = ["Pallidum"]  # Regions to exclude
RLD = False  # Reload regressor matrices instead of computing them again

print("\nRELOADING REGRESSORS!\n") if RLD else ...

#raise

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

# Choose variables
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [diab, age, sex, college, age_onset]
        ) \
        .drop("age_onset", axis=1)

# If contrast is age
if CTRS == "age":
    # Exclude subjects with T2DM
    regressors = regressors.query("diab == 0")

# Inner merge regressors with a gm column to make sure all included subjects have data
y = data['Volume of grey matter (normalised for head size)'].rename("feat").reset_index()
regressors_y = y.merge(regressors, on="eid", how="inner")

## Fit model
#sdf = regressors_y

#model = smf.ols(f"feat ~ C(diab) + age + C(sex) + C(college)", data=sdf)
#results = model.fit()
#
## Print results
#results.summary()

## Check assumptions
#check_assumptions(results, sdf)

# Drop feat column
regressors_clean = regressors_y.drop(["feat"], axis=1)

# Save full regressor matrix
regressors_clean.to_csv(OUTDIR + f"regressors/pub_meta_volume_full_regressors_{CTRS}.csv")

if CTRS == "age":

    # Interactions among independent variables
    var_dict = {
            "sex": "disc",
            "college": "disc",
            }

    for name, type_ in var_dict.items():

        check_covariance(
                regressors_clean,
                var1=name,
                var2="age",
                type1=type_,
                type2="cont",
                save=True,
                prefix=OUTDIR + "covariance/pub_meta_volume_covar"
                )

        plt.close("all")

    if RLD == False:
        # Match
        regressors_matched = match_cont(
                df=regressors_clean,
                main_var="age",
                vars_to_match=["sex", "college"],
                N=1,
                random_state=1
                )

if CTRS == "diab":

    # Interactions among independent variables
    var_dict = {
            "age": "cont",
            "sex": "disc",
            "college": "disc",
            }

    for name, type_ in var_dict.items():

        check_covariance(
                regressors_clean,
                var1="diab",
                var2=name,
                type1="disc",
                type2=type_,
                save=True,
                prefix=OUTDIR + "covariance/pub_meta_volume_covar"
                )

        plt.close("all")

    if RLD == False:
        # Match
        regressors_matched = match_mah(
                df=regressors_clean,
                main_var="diab",
                vars_to_match=["age", "sex", "college"],
                N=1,
                random_state=1
                )

if RLD == False:
    # Save matched regressors matrix
    regressors_matched.to_csv(OUTDIR + f"regressors/pub_meta_volume_matched_regressors_{CTRS}.csv")

# %%
# =============================================================================
# Extract gray matter volumes for regions from raw data
# =============================================================================

# Status
print(f"Extracting volume values with contrast [{CTRS}] at parcellation [{PARC}]")

# Load regressors
regressors_matched = pd.read_csv(
        OUTDIR + f"regressors/pub_meta_volume_matched_regressors_{CTRS}.csv"
        )

# 46 parcellation
# -------
if PARC == 46:

    # Parcellation data
    parc_data = pd.read_csv(SRCDIR + "atlas/46/139_to_46_indexes.csv", index_col=0)

    # Make a code book
    col_names = np.array(data.columns)

    # New dataframe for merged features
    data_extracted = pd.DataFrame()

    # Iterate over all groups of regions
    for item in parc_data.iterrows():

        # Indexes belonging to current group of regions
        indexes = [int(val)+24 for val in item[1]["index"][1:-1].split(", ")]

        # Get corresponding labels
        current_col_names = col_names[indexes]

        # Name of the current group of regions
        name = item[1]["label"]

        # Merge and construct current group of regions
        current_group = data[current_col_names].dropna(how="any").sum(axis=1).rename(name)

        # Normalize with head size
        data_extracted[name] = current_group/(head_norm.set_index("eid")["norm_fact"])

#        print(name, current_col_names)

    # Add regressors to data
    df = data_extracted.merge(regressors_matched, on="eid", how="inner")

    # Extract list of features from new feat groups
    features = list(data_extracted.columns)

    # Exclude regions that are too small
    features = [feat for feat in features if feat not in excl_region]

# 139 parcellation
# -----
if PARC == 139:

    # Parcellation data
    parc_data = pd \
        .read_csv(SRCDIR + "atlas/139/ukb_gm_labelmask_139.csv") \
        .pipe(lambda df:
            df.assign(**{
              "label": df["label"].apply(lambda item: item \
                             .replace(" ", "_") \
                             .replace(",", "") \
                             .replace("'", "") \
                             .replace("-", ""))
              })) \

    # Assign labels to features
    features = parc_data["label"].to_list()

    # Transform raw gray matter volume data
    df = data \
        .set_axis(list(data.columns[:25]) + features, axis=1) \
        .iloc[:, 25:] \
        .divide(head_norm.set_index("eid")["norm_fact"], axis="index") \
        .merge(regressors_matched, on="eid", how="inner")


# %%
# =============================================================================
# Fit models
# =============================================================================

# Status
print(f"Fitting models for contrast [{CTRS}] at parcellation [{PARC}]")

# Dictionary to store stats
feat_stats = {}

# Iterate over all regions
for i, feat in tqdm(enumerate(features), total=len(features), desc="Models fitted: "):

    # Prep
    # ----
    # Extract current feature
    sdf = df[["eid", "age", "diab", "sex", "college", f"{feat}"]]

    # Get sample sizes
    sample_sizes = sdf.groupby("diab")["eid"].count()

    # Fit
    # -----
    # Formula
    if CTRS == "age":
        formula = f"{feat} ~ age + C(sex) + C(college)"
    if CTRS == "diab":
        formula = f"{feat} ~ C(diab) + age + C(sex) + C(college)"

    # Fit model
    model = smf.ols(formula, data=sdf)
    results = model.fit()

    # Monitor
    # --------

    # Save detailed stats report
    with open(OUTDIR + f"stats_misc/pub_meta_volume_regression_table_{feat}" \
              f"_{CTRS}_{PARC}.html", "w") as f:
        f.write(results.summary().as_html())

    # Check assumptions
    check_assumptions(
            results,
            sdf,
            prefix=OUTDIR + \
            f"stats_misc/pub_meta_volume_stats_assumptions_{feat}_{CTRS}_{PARC}"
            )

    # Plot across age
    if CTRS == "diab":
        gdf = sdf \
            [[feat, "age", "diab"]] \
            .pipe(lambda df: df.assign(**{"age_group":
                    pd.cut(df["age"], bins=np.arange(0, 100, 5)).astype(str)
                    })) \
            .sort_values(by="age")

        plt.figure(figsize=(10, 7))
        plt.title(feat)
        sns.lineplot(data=gdf, x="age_group", y=feat, hue="diab",
                     palette=sns.color_palette(["black", "red"]),
                     ci=68, err_style="bars",
                     marker="o", linewidth=2, markersize=6,
                     err_kws={"capsize": 3, "capthick": 2, "elinewidth": 2})
        plt.tight_layout()
        plt.savefig(OUTDIR + f"stats_misc/pub_meta_volume_age-diab-plot_{feat}_{PARC}.pdf")
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
    beta = results.params.loc[rel_key] \
        /norm_fact

    # Get confidence intervals
    conf_int = results.conf_int().loc[rel_key, :]/norm_fact

    # Get p value
    pval = results.pvalues.loc[rel_key]

    # Save stats as dict
    feat_stats[f"{feat}"] = [list(sample_sizes), tval, pval, beta,
                              np.array(conf_int)]


# Convert stats to df and correct p values for multicomp
df_out = pd.DataFrame.from_dict(
        feat_stats, orient="index",
        columns=["sample_sizes", "tval", "pval", "beta", "conf_int"]) \
        .reset_index() \
        .rename({"index": "label"}, axis=1) \
        .assign(**{"pval": lambda df: pg.multicomp(list(df["pval"]),
                                                   method="bonf")[1]})

# Save outputs
df_out.to_csv(OUTDIR + f"stats/pub_meta_volume_stats_{CTRS}_{PARC}.csv")
