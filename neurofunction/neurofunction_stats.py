#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 00:22:55 2021

@author: botond

Notes:
-this script performs linear regression on functional MRI results (ALFF)
in voxel space

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
get_ipython().run_line_magic('cd', 'neurofunction')


# =============================================================================
# Setup
# =============================================================================

plt.style.use("ggplot")

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/neurofunction/"

# Inputs
CTRS = "age"  # Contrast: diab or age
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
excl_sub = [] # [1653701, 3361084, 3828231, 2010790, 2925838, 3846337,]  # Subjects
## to exlucde due to abnormal total gray matter neurofunctions
RLD = False  # Reload regressor matrices instead of computing them again

#raise

# %%
# =============================================================================
# Load data
# =============================================================================

# Load batch
# -------
img_subs = pd.read_csv(SRCDIR + "neurofunction/pub_meta_subjects_batch7_alff.csv",
                       index_col=0)

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

# =============================================================================
# Build regressor matrix
# =============================================================================

# Choose variables
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [diab, age, sex, college, ses, bmi]
        )

# If contrast is age
if CTRS == "age":
    # Exclude subjects with T2DM
    regressors = regressors.query("diab == 0")

# Inner merge regressors with subjects for whom ALFF has been computed
regressors_clean = regressors.merge(img_subs, on="eid")

# Let's see what matching would look like

# Save full regressor matrix
regressors_clean.to_csv(OUTDIR + f"regressors/pub_meta_neurofunction_full_regressors_{CTRS}.csv")

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
                prefix=OUTDIR + "covariance/pub_meta_neurofunction_covar_"
                )

        plt.close("all")

    if RLD == False:
        # Match
        regressors_matched = match(
                df=regressors_clean,
                main_var="sex",
                vars_to_match=["age"],
                N=1,
                random_state=1
                )

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
                prefix=OUTDIR + "covariance/pub_meta_neurofunction_covar_"
                )

        plt.close("all")

    if RLD == False:
        # Match
        regressors_matched = match(
                df=regressors_clean,
                main_var="diab",
                vars_to_match=["age", "sex"],
                N=1,
                random_state=1
                )

if RLD == False:
    # Save matched regressors matrix
    regressors_matched.to_csv(OUTDIR + f"regressors/pub_meta_neurofunction_matched_regressors_{CTRS}.csv")




#TODO: add in functional stats script here, from the point where the design matrix \
# is computed (just rename subject column)
