#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:02:05 2021

@author: botond
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 12:57:25 2021

@author: botond

The goal is to quantify similarity along the following factors:

T2DM - aging
UKB - meta
Volumetric - neurofunctional measurements

Meta only contains 1 type (mix of functional and structural). So that reduces
the number of possible comparisons. Makes it 3+4+2 in total:

1. age vs T2DM:
    ukb func
    ukb atrophy
    neuroquery
2. UKB vs meta:
    age: ukb func vs meta
    age: ukb atrophy vs meta
    t2dm: ukb func vs meta
    t2dm: ukb atrophy  vs meta
3. atrophy vs functional:
    ukb t2dm
    ukb age

Decisions:
46 vs 139 parcellation
mean vs max vs etc
spearman vs pearson vs other


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
get_ipython().run_line_magic('cd', 'neurofunction')
get_ipython().run_line_magic('matplotlib', 'auto')

# %%
# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "results/"
OUTDIR = HOMEDIR + "results/similarity/"

PC = 46  # Parcellation version (46 or 139)
CORRMET = "spearman"
EXTRA = "_mean"

contrasts = ["age", "diab"]
modalities = ["volume", "neurofunction", "neuroquery"]
ds_extra = "alff_batch7_GM_0.5"
vol_field = "beta"  # Numerical field to consider from volumetric data

gm_thr = 0.5  # GM threshold

#raise
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
for ct in contrasts:
    data_list[f"volume_{ct}"] = \
        pd \
            .read_csv(SRCDIR + f"volume/stats/pub_meta_volume_stats_{ct}_{PC}.csv",
                      index_col=None) \
            .pipe(lambda df:
                df.assign(**{
                        "index": np.arange(df.shape[0]) + 1,
                        "label": df["label"],
                        "value": df[vol_field]
                        })) \
            [["index", "label", "value"]]

#"label": df["label"].apply(lambda item: (" ").join(item.split("_"))),

# Import functional data
# -----
# All possible combos, some do not exist
combos = itertools.product(contrasts, modalities)

# Load the ones that exist
for combo in combos:
    ct, mod = combo
    if mod == "neurofunction":
        ext = ds_extra + "_"
    else:
        ext = ""

    filename = f"pub_meta_{mod}_zmap_{ext}contrast_{ct}.nii"
    try:
        img = image.load_img(SRCDIR + f"{mod}/stats/" + filename)
        raw_list[f"{mod}_{ct}"] = img
        print(f"Loaded {filename}")
    except Exception as e:
        pass
#        print(f"Did not load {filename}", e)

# Status
print("Loading parcellation.")

# Load anatomical atlas
# -----
# Label mask
label_mask = image.load_img(HOMEDIR + f"data/atlas/{PC}/ukb_gm_labelmask_{PC}.nii.gz")

# Label list
label_list = pd.read_csv(HOMEDIR + f"data/atlas/{PC}/ukb_gm_labelmask_{PC}.csv") \
    .pipe(lambda df:
        df.assign(**{
                "label": df["label"].apply(
                        lambda item: item.replace(" ", "_")
                        )})
    )

# Load gray matter mask
# ---------
# Load mask
gm_img = image.load_img(HOMEDIR + "data/atlas/" \
                        "mni_icbm152_gm_tal_nlin_asym_09c.nii")

# Resample gm_mask
gm_img_rs = image.resample_img(
            gm_img,
            target_affine=label_mask.affine,
            target_shape=label_mask.shape,
            interpolation="nearest"
                )

# Binarize gm mask
gm_mask = image.math_img(f'img > {gm_thr}', img=gm_img_rs)

# %%
# =============================================================================
# Convert functional data
# =============================================================================

# Build mask
# -----

# Status
print("Building mask.")

# Dictionary to store masks
label_masks = {}

# Iterate over all regions
for item in tqdm(list(label_list.itertuples())):

    # Extract ROI's label and value form current row
    label, value = item.label, item.value

    # Build mask
    label_masks[value] = image.math_img(
                                f"np.where(img == {value}, 1, 0)",
                                img=label_mask,
                                        )

# %%
# Apply masks and coarsen
# ------

# Status
print("Applying masks.")

def coarsen_effect(file):
    """
    Quantifies relevance of effects to each anatomical label.
    """

    # Get img file
    img = raw_list[file]

    # Resample img
    img = image.resample_img(
            img,
            target_affine=label_mask.affine,
            target_shape=label_mask.shape
            )

    # Apply gm mask if needed (UKB is already masked)
    if "neuroquery" in file:
        img = image.threshold_img(img, threshold="0%", mask_img=gm_mask)

    # List of coverages for current img
    df_list = []

    # Iterate over all regions
    for item in tqdm(list(label_list.itertuples()), desc=file):

        # Extract ROI's label and value form current row
        label, value = item.label, item.value

        # Mask
        mask_img = label_masks[value]

        # Apply mask
        masker = input_data.NiftiMasker(mask_img)
        masked_img = masker.fit_transform(img)

        # Apply operation
        # A: mean
        # =======
        res = np.mean(masked_img)

#        # B: max/min
#        # =======
#        # Take extremes both ways
#        val_max = np.max(masked_img)
#        val_min = np.min(masked_img)
#
#        # Pick the most extreme by absolute value
#        res = [val_max, val_min][np.argmax(np.abs([val_max, val_min]))]

        # ========
        # Store
        df_list.append(pd.DataFrame([value, label, res]) \
                       .T.set_axis(["index", "label", "value"], axis=1) \
                       .astype({"value": float}))

    df = pd.concat(df_list, axis=0)

    return df

# Iterate over all raw functional files
for file in raw_list.keys():

    # Status
#    print(file)

    # Coarsen
    data_list[file] = coarsen_effect(file)

# %%
# Merge dataframes
# -----

# Status
print("Merging Dataframes.")

keys = list(data_list.keys())

# Custom order
keys = [
     'volume_age',
     'volume_diab',
     'neurofunction_age',
     'neurofunction_diab',
     'neuroquery_age',
     'neuroquery_diab'
     ]

# Helper functions
unique_col = lambda key: data_list[key].rename({"value": key}, axis=1)
merge = lambda a, b: a.merge(b, on=["index", "label"], how="inner")

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
corr_matrix = df.set_index(["index", "label"]).corr(method=CORRMET)

# Compute staistical significance of correlations
def comp_corr_pvals(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = eval(f"stats.{CORRMET}r(df[r], df[c])")[1]
    return pvalues.astype(float)

corr_pvals_raw = comp_corr_pvals(df)
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
plt.figure(figsize=(12, 10))
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.title(f"Parcellation: {PC}, {CORRMET}, {EXTRA}")
sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="seismic", annot=annot_df,
            fmt="", linewidth=1, linecolor="k",
            annot_kws={"fontsize": 16})
plt.xticks(rotation=45, ha="right");
plt.tight_layout()
plt.savefig(OUTDIR + f"figures/corr_matrix_{PC}_{CORRMET}{EXTRA}.pdf")

# TODO
## Tidy ip the figure
## --------
#plt.title("")
#plt.xlabel(ticktext)
#
#plt.savefig(OUTDIR + f"figures/corr_matrix_{PC}_{CORRMET}{EXTRA}.pdf")

# %%
# Scatterplot
'''
'ukb_volume_age'
'ukb_func_alff_age'
'neuroquery_mix_age'
'ukb_volume_diabetes'
'ukb_func_alff_diabetes'
'neuroquery_mix_diabetes'
'''

df.columns = list(map(lambda x: x.replace(" ", "_"), list(df.columns)))

combos = list(itertools.combinations(keys, 2))

for combo in tqdm(combos):
    a, b = combo
    plt.figure()
    plt.scatter(df[a], df[b])
#    for i in range(len(df)):
#        plt.annotate(df.loc[i, "label"], xy=[df.loc[i, a], df.loc[i, b]])
    plt.xlabel(a)
    plt.ylabel(b)
    plt.tight_layout()
    plt.savefig(OUTDIR + f"scatterplots/scatter_{PC}{EXTRA}_{a}_{b}.pdf")
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
