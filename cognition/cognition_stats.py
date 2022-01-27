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
from helpers.regression_helpers import check_covariance, match, match_cont, check_assumptions
from helpers.data_loader import DataLoader
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
T1DM_CO = 40  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
AGE_CO = 50  # Age cutoff (related to T1DM_CO) to avoid T2DM low duration subjects
STRAT_SEX = 1 # Stratify sex or not #TODO: need to adjust detrending accordinlgy
SEX = 0  # If stratifying per sex, which sex to keep

EXTRA = "_F"  # Extra suffix for saved files
RLD = False  # Reload regressor matrices instead of computing them again

print("\nRELOADING REGRESSORS!\n") if RLD else ...

# <><><><><><><><>
# raise
# <><><><><><><><>

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

# Initiate loader object
dl = DataLoader()

# Load data
dl.load_basic_vars(SRCDIR)

# Extract relevant variables
age, sex, diab, college, bmi, mp, hrt, age_onset, duration, htn = \
    (dl.age, dl.sex, dl.diab, dl.college, dl.bmi, dl.mp, dl.hrt, dl.age_onset, \
    dl.duration, dl.htn)


# Restrictive variables
# -----

# Perform filtering
dl.filter_vars(AGE_CO, T1DM_CO)

# Extract filtered series
age, mp, hrt, age_onset = dl.age, dl.mp, dl.hrt, dl.age_onset


# %%
# =============================================================================
# Build regressor matrices
# =============================================================================

# Status
print(f"Building regressor matrix with contrast [{CTRS}]")

# Merge IVs and put previously defined exclusions into action (through merge)
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [age, sex, college, diab, mp, hrt, htn, age_onset]
        ) \
        .drop(["mp", "hrt", "age_onset"], axis=1)

# Sample sizes
#regressors.merge(data, on="eid").set_index(list(regressors.columns)) \
#        .pipe(lambda df: df[df>=0]).dropna(how="all").mean(axis=1).groupby("diab").count()

# If contrast is age
if CTRS == "age":
    # Exclude subjects with T2DM
    regressors = regressors.query("diab == 0")

# If contrast is sex and we want to separate across age
if CTRS == "sex":
    # Exclude subjects with T2DM OR subjects without T2DM (toggle switch)
    regressors = regressors.query("diab == 0")

# Optional: stratify per sex
if STRAT_SEX:
    # Include subjects of one sex only
    regressors = regressors.query(f"sex == {SEX}")

# Assign labels to features
features = [col for col in data.columns if "eid" not in col]

# Iterate over all features
for i, feat in enumerate(features):

    # Status
    print(f"Current feature: {feat}")

    # Clean feature
    # -----
    # Extract data, drop non-positive records
    data_feat = data.query(f'`{feat}` > 0')[["eid", feat]]

    # Constrain ivs to only those samples that have y values
    regressors_y = regressors \
        .merge(data_feat, on="eid", how="inner") \
        .dropna(how="any") \

    # Temporarily remove y values
    regressors_clean = regressors_y.drop(feat, axis=1)

    # Check covariance of ivs
    # -----

    if CTRS == "age":

        # Interactions among independent variables
        var_dict = {
                "sex": "disc",
                "college": "disc",
                }

        # for name, type_ in var_dict.items():

        #     check_covariance(
        #             regressors_clean,
        #             var1=name,
        #             var2="age",
        #             type1=type_,
        #             type2="cont",
        #             save=True,
        #             prefix=OUTDIR + f"covariance/pub_meta_cognition_covar_{feat}"
        #             )

        #     plt.close("all")

    if CTRS == "diab":

        # Interactions among independent variables
        var_dict = {
                "age": "cont",
                "sex": "disc",
                "college": "disc",
                }

        # for name, type_ in var_dict.items():

        #     check_covariance(
        #             regressors_clean,
        #             var1="diab",
        #             var2=name,
        #             type1="disc",
        #             type2=type_,
        #             save=True,
        #             prefix=OUTDIR + f"covariance/pub_meta_cognition_covar_{feat}"
        #             )

        #     plt.close("all")


    if CTRS == "sex":

        # Interactions among independent variables
        var_dict = {
                "age": "cont",
                "college": "disc",
                "htn": "disc"
                }

        # for name, type_ in var_dict.items():

        #     check_covariance(
        #             regressors_clean,
        #             var1="sex",
        #             var2=name,
        #             type1="disc",
        #             type2=type_,
        #             save=True,
        #             prefix=OUTDIR + "covariance/pub_meta_cognition_covar"
        #             )

        #     plt.close("all")

    # Perform matching
    # ------
    if (CTRS == "age") & (RLD == False):

        # Match
        regressors_matched = match_cont(
                df=regressors_clean,
                main_vars=["age"],
                vars_to_match=["sex", "college", "htn"],
                value=3,
                random_state=111
                )

    if (CTRS == "diab") & (RLD == False):

        # Match
        regressors_matched = match(
            df=regressors_clean,
            main_vars=["diab"],
            vars_to_match=["age", "sex", "college", "htn"],
            random_state=111 # 9,10,111 with bmi
            )

    if (CTRS == "sex") & (RLD == False):

        # Match
        regressors_matched = match(
                df=regressors_clean,
                main_vars=["sex"],
                vars_to_match=["age", "college", "htn"],
                random_state=10
                )

    # Save matched regressors matrix
    if RLD == False:
        regressors_matched.to_csv(OUTDIR + \
            f"regressors/pub_meta_cognition_matched_regressors_{feat}_{CTRS}{EXTRA}.csv")

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
            f"regressors/pub_meta_cognition_matched_regressors_{feat}_{CTRS}{EXTRA}.csv",
            index_col=0)

    # Reunite regressors and y values of the current feature
    sdf = regressors_matched.merge(data_feat, on="eid")

    # Get sample sizes
    sample_sizes = sdf.groupby("sex" if CTRS == "sex" else "diab")["eid"].count()

    # Regression
    # -------
    # Formula
    if CTRS == "age":
        formula = f"{feat} ~ age + C(sex) + C(college) + C(htn)"
    if CTRS == "diab":
        formula = f"{feat} ~ C(diab) + age + C(sex) + C(college) + C(htn)"
    if CTRS == "sex":
        formula = f"{feat} ~ age + C(sex) + C(college) + C(htn)"

    # Fit model
    model = smf.ols(formula, data=sdf)
    results = model.fit()

    # Monitor
    # ------
    # Save detailed stats report
    with open(OUTDIR + f"stats_misc/pub_meta_cognition_regression_table_{feat}" \
              f"_{CTRS}{EXTRA}.html", "w") as f:
        f.write(results.summary().as_html())

    # Check assumptions
    check_assumptions(
            results,
            sdf,
            prefix=OUTDIR + \
            f"stats_misc/pub_meta_cognition_stats_assumptions_{feat}_{CTRS}{EXTRA}"
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
        plt.savefig(OUTDIR + f"stats_misc/pub_meta_cognition_age-diab-plot_{feat}{EXTRA}.pdf")
        plt.close()

    # Plot distribution of feature
    plt.figure()
    plt.title(feat)
    plt.hist(sdf[feat])
    plt.xlabel("score")
    plt.ylabel("count")
    plt.savefig(OUTDIR + f"stats_misc/pub_meta_cognition_dist_{feat}_{CTRS}{EXTRA}.pdf")
    plt.tight_layout()
    plt.close()

    # Save results
    # -------
    # Normalization factor
    if CTRS in ["age", "diab"]:
        norm_fact = sdf.query('diab==0')[feat].mean()/100

    elif CTRS == "sex":
        norm_fact = sdf.query('sex==0')[feat].mean()/100

    # Get relevant key for regressor
    rel_key = [key for key in results.conf_int().index.to_list() \
           if CTRS in key][0]

    # Get effect size
    tval = results.tvalues.loc[rel_key]
    beta = results.params.loc[rel_key]/norm_fact

    # Get confidence intervals
    conf_int = results.conf_int().loc[rel_key, :]/norm_fact
    plus_minus = beta - conf_int[0]

    # Get p value
    pval = results.pvalues.loc[rel_key]

    # Save stats as dict
    feat_stats[f"{feat}"] = [list(sample_sizes), tval, pval, beta,
                             np.array(conf_int), plus_minus]


# Convert stats to df and correct p values for multicomp
df_out = pd.DataFrame.from_dict(
        feat_stats, orient="index",
        columns=["sample_sizes", "tval", "pval", "beta", "conf_int", "plus_minus"]) \
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
df_out.to_csv(OUTDIR + f"stats/pub_meta_cognition_stats_{CTRS}{EXTRA}.csv")
