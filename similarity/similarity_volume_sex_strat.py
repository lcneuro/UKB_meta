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

for combo in tqdm(combos):
    a, b = combo
    # plt.figure(figsize=(3.5, 3.5))
    # plt.scatter(df[a], df[b])
    # for i in range(len(df)):
    #     plt.annotate(df.loc[i, "label"], xy=[df.loc[i, a], df.loc[i, b]])
    sns.lmplot(data=df, x=a, y=b, height=3.5,
               line_kws={"linewidth": 2, "zorder": 2},
               scatter_kws={"linewidth": 0.7, "edgecolor": "k", "zorder": 3})
    plt.xlabel(a.replace("_", " "))
    plt.ylabel(b.replace("_", " "))
    plt.tight_layout()
    plt.savefig(OUTDIR + f"scatterplots/scatter_{PC}_strat{EXTRA}_{a}_{b}.pdf")
    plt.close("all")

plt.close("all")

# %% Testing ground
#temp = nib.load(WORKDIR + "age.nii.gz")
#nib.save(temp, WORKDIR + "neuroquery_mix_age.nii")
#
#test = image.threshold_img(img, threshold=-999999, mask_img=gm_mask)
#
##test = image.math_img(f'img*
#
#plotting.view_img(test, bg_img=False).open_in_browser()

#mylist = []
#for file, img in raw_list.items():
#    # Resample img
#    img = image.resample_img(
#            img,
#            target_affine=label_mask.affine,
#            target_shape=label_mask.shape
#            )
#
#    # Apply gm mask if needed (UKB is already masked)
#    dat = masking.apply_mask(img, gm_mask)
#
#    mylist.append(pd.DataFrame(dat.flatten(), columns=[file]))
#
#tempdf = pd.concat(mylist, axis=1)
#tempcorr = tempdf.corr(method="spearman")
#
#plt.hist(tempdf.iloc[:, 0])
