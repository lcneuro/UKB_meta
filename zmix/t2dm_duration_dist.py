#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 23:47:10 2021

@author: botond

Notes:
-this script looks at how disease duration is distributed across the life span
among subjects with T2DM

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



# =============================================================================
# Setup
# =============================================================================
plt.style.use("ggplot")
sns.set_style("whitegrid")

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/"

# Inputs
src = "cognition/regressors/" \
             "pub_meta_cognition_matched_regressors_Reaction_Time_diab.csv"

#raise

# %%
# =============================================================================
#  Load regressors
# =============================================================================

# Load regressor matrix for specific case
regressors = pd.read_csv(OUTDIR + src, index_col=0)

# Age of diabetes diagnosis (rough estimate!, averaged)
age_onset = pd \
    .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
    .set_index("eid") \
    .mean(axis=1) \
    .rename("age_onset") \
    .reset_index()

# %%
# =============================================================================
#  Check relationship
# =============================================================================
# Transform
df = regressors \
    .query("diab == 1") \
    .merge(age_onset, on="eid") \
    .dropna() \
    .pipe(lambda df:
        df.assign(**{"duration": df["age"] - df["age_onset"]
        }))

# Linear stats
corr = stats.pearsonr(df["age"], df["duration"])

# Plot
sns.lineplot(data=df, x="age", y="duration")

# Format
plt.title("T2DM Duration vs. Age")
text = f"Pearson's r: r={corr[0]:.3f}, p={corr[1]:.2e}, n={df.shape[0]}" + \
        f"\nsource file: {src[:40]}\n{src[40:]}"
plt.annotate(text, xy=[0.05, 0.85], xycoords="axes fraction")
plt.tight_layout()

# Save
plt.savefig(OUTDIR + "zmix/pub_meta_t2dm-duration-age.pdf")

# Close
plt.close()
