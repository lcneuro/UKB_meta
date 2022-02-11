#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 12:57:25 2021

@author: botond

The goal is to quantify similarity along the following factors:

T2DM - aging
F - M
(Volumetric only)


"""

import os
import itertools
import functools
import numpy as np
import pandas as pd
from nilearn import image, input_data, masking
import nibabel as nib
from scipy import stats, special
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from adjustText import adjust_text
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'similarity')
get_ipython().run_line_magic('matplotlib', 'inline')

# %%
# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "results/"
OUTDIR = HOMEDIR + "results/similarity/"

PC = 46  # Parcellation version (46 or 139)
CORRMET = "pearson"
EXTRA = "_mean"

contrasts = ["age", "diab"]
strats = ["F", "M"]
modalities = ["volume"]
vol_field = "beta"  # Numerical field to consider from volumetric data

# <><><><><><><><>
# raise
# <><><><><><><><>

# %%
# =============================================================================
# Import data
# =============================================================================
"""
Going to put everything into a flat list
"""

# Status
print("Loading data.")

# Lists to store data
data_list = {}  # Final list for deriving similarities
raw_list = {}  # Temp list for storing raw functional images

# Import UKB atrophy
# -----
for ct, st in itertools.product(contrasts, strats):

    print(ct, st)

    data_list[f"volume_{ct}_{st}"] = \
        pd \
            .read_csv(SRCDIR + f"volume/stats/pub_meta_volume_stats_{ct}_{PC}_{st}.csv",
                      index_col=None) \
            .pipe(lambda df:
                df.assign(**{
                        "label": df["label"],
                        "value": df[vol_field]
                        })) \
            [["label", "value"]]

#"label": df["label"].apply(lambda item: (" ").join(item.split("_"))),

# %%
# Merge dataframes
# -----

# Status
print("Merging Dataframes.")

keys = list(data_list.keys())

# Custom order
keys = [
     'volume_age_F',
     'volume_age_M',
     'volume_diab_F',
     'volume_diab_M',
     ]

# Helper functions
unique_col = lambda key: data_list[key].rename({"value": key}, axis=1)
merge = lambda a, b: a.merge(b, on=["label"], how="inner")

# Merge dfs
df = functools.reduce(merge, list(map(unique_col, keys)))

# Reformat column names
df.columns = list(map(lambda x: x.replace("_", " "), list(df.columns)))

# Save for later
#df.to_csv(OUTDIR + f"df_{PC}{EXTRA}.csv")
#df.info()

# Open
#df = pd.read_csv(OUTDIR + f"df_.csv")

# %%
# =============================================================================
# Derive similarities
# =============================================================================

# Compute correlations
#corr_matrix = df.set_index(["index", "label"]).corr(method=CORRMET)

# Compute staistical significance of correlations
def comp_corr_pvals(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    corr_matrix = dfcols.transpose().join(dfcols, how='outer')
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            corr_matrix[r][c], pvalues[r][c] = eval(f"stats.{CORRMET}r(df[r], df[c])")
    return [corr_matrix.astype(float), pvalues.astype(float)]

corr_matrix, corr_pvals_raw = comp_corr_pvals(df)
# Not bonferroni corrected yet! Correction takes place below!

# Perform Bonferroni correction
# n=number of elements in one of the triangles
corr_pvals = corr_pvals_raw * special.comb(corr_pvals_raw.shape[0], 2)

# %%
# =============================================================================
# Visualize
# =============================================================================

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

# Annotation
rhos = corr_matrix.values.flatten()
pvals = corr_pvals.values.flatten()
annot_text = [f"{rhos[i]:.2f}\n" + p2star(pvals[i]) for i in range(len(rhos))]
annot_df = pd.DataFrame(np.array(annot_text).reshape(corr_matrix.shape))

# Correlation plot
plt.figure(figsize=(4.8, 4.2))
#plt.rcParams['xtick.labelsize']=16
#plt.rcParams['ytick.labelsize']=16
plt.title(f"Correlation Based Similarities between\nGray Matter Volumetric " \
          "Effects Associated with\nAge and T2DM, Quantified Separately for Sexes",
          )
g = sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="seismic", annot=annot_df,
            fmt="", linewidth=1, linecolor="k",
            annot_kws={"fontsize": 8*fs})
g.figure.axes[-1].tick_params(labelsize=6*fs)
plt.xticks(rotation=45, ha="right");
plt.tight_layout()
plt.savefig(OUTDIR + f"figures/corr_matrix_{PC}_volume_strat{EXTRA}.pdf")



# TODO
## Tidy ip the figure
## --------
# plt.title("")
# plt.xlabel(ticktext)
#
# plt.savefig(OUTDIR + f"figures/corr_matrix_{PC}_{CORRMET}{EXTRA}.pdf")

# %%
# Scatterplot
'''
'ukb_volume_age_F'
'ukb_volume_age_M'
'ukb_volume_diab_F'
'ukb_volume_diab_M'
'''

df.columns = list(map(lambda x: x.replace(" ", "_"), list(df.columns)))

combos = list(itertools.combinations(keys, 2))
icombos = list(itertools.combinations(np.arange(df.shape[1]-1), 2))

# It over
for c, combo in tqdm(enumerate(combos)):

    # Unpack label combo
    a, b = combo

    # Unpack corresponding indexes
    i1, i2 = icombos[c]

    # Seaborn lmplot
    sns.lmplot(data=df, x=a, y=b, height=3.5,
               line_kws={"linewidth": 2, "zorder": 2},
               scatter_kws={"s": 15, "linewidth": 0.7, "edgecolor": "k", "zorder": 3})

    # Optional: annotate labels to points >
    # -----
    # Decide which labels to annotate based on coords, to avoid overlapping text
    # Based on x coord
    # limits_x = np.percentile(np.linspace(df.loc[:, a].min(), df.loc[:, a].max(), 101), [2, 98])
    limits_x = np.percentile(df.loc[:, a], [3, 97])

    # Based on y coord
    # limits_y = np.percentile(np.linspace(df.loc[:, b].min(), df.loc[:, b].max(), 101), [2, 98])
    limits_y = np.percentile(df.loc[:, b], [1, 97])

    # Based on how much it drives the correlation
    pearson_terms = (df.loc[:, a] - df.loc[:, a].mean()) * (df.loc[:, b] - df.loc[:, b].mean())
    # limits_p = np.percentile(np.linspace(pearson_terms.min(), pearson_terms.max()), [10, 80])
    limits_p = np.percentile(pearson_terms, [5, 95])

    cond_x = (df.loc[:, a] < limits_x[0]) | (df.loc[:, a] > limits_x[1])
    cond_y = (df.loc[:, b] < limits_y[0]) | (df.loc[:, b] > limits_y[1])
    cond_p = (pearson_terms < limits_p[0]) | (pearson_terms > limits_p[1])

    # Final mask
    annot = cond_x | cond_y #| cond_p

    # List of annotations for later adjustment
    annot_list = []

    # Annotate labels
    for i in range(len(df)):
        if annot[i]:
            annot_list.append(
                            plt.annotate(
                                df.loc[i, "label"].replace("_", " "),
                                xy=[df.loc[i, a], df.loc[i, b]],
                                        )
                            )

    # Adjust labels to avoid overlap
    adjust_text(annot_list, arrowprops=dict(arrowstyle="-", linewidth=1,
                                            connectionstyle="angle3"),
                expand_points=[1.5, 3])

    # -----

    # Annotate stats to fig
    text = f"r = {corr_matrix.iloc[i1, i2]:.2f}\n{pformat(corr_pvals.iloc[i1, i2])}"
    plt.annotate(text, xy=[0.05, 0.9], xycoords="axes fraction",
                 bbox=dict(boxstyle='square', fc='white'))

    # Formatting
    plt.xlabel(a.replace("_", " @").replace("diab", "T2DM") \
               .replace("@M", "(Males)").replace("@F", "(Females)"))
    plt.ylabel(b.replace("_", " @").replace("diab", "T2DM") \
               .replace("@M", "(Males)").replace("@F", "(Females)"))

    xlim = np.array(plt.gca().get_xlim())
    xpad = 0.05*(xlim[1] - xlim[0])
    plt.xlim(xlim[0] - xpad, xlim[1] + xpad)

    plt.tight_layout()
    plt.savefig(OUTDIR + f"scatterplots/scatter_{PC}_strat{EXTRA}_{a}_{b}.pdf")
    plt.close("all")

plt.close("all")

