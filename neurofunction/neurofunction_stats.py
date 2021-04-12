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

from pprint import pprint
from nilearn import image
from nilearn import masking
from nilearn import input_data
from nilearn import plotting
from nistats.design_matrix import make_second_level_design_matrix
from nistats.second_level_model import SecondLevelModel
from nistats.reporting import plot_design_matrix
from nistats.thresholding import map_threshold
import nibabel as nib

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match, match_cont, check_assumptions
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
RLD = False  # Reload regressor matrices instead of computing them again
BATCH = 7  # Batch of preprocessed images to use

T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
## to exlucde due to abnormal total gray matter neurofunctions

CTRS = "diab"  # Contrast: diab or age
CT = 12  # Cluster threshold for group level analysis (in voxels?)
PTHR = 0.05  # Significance threshold for p values
GM_THR = 0.5  # Threshold for gm mask. Voxels with Probability above this
MDL = "ALFF"  # Modality
EXTRA = ""

VOLUME = 1000  # Cluster volume, mm3
VOXEL_DIM = 2.4  # Voxel dimension in work space, assuming cubic voxels, mm

regressors_dict = {
    "diab": ["subject_label", "diab", "age", "sex", "college"],
    "age": ["subject_label", "age", "sex", "college"],
    }

print("Settings:\n" \
    f"CONTRAST={CTRS}, MDL={MDL}, PTHR={PTHR}, CT={CT}")

#raise

# %%
# =============================================================================
# Load data
# =============================================================================

# Status
print("Loading data.")

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
# Create gm mask
# =============================================================================

# Status
print("Preparing gm mask.")

# Load target affine and target shape (ignore values except for shape and affine)
target_img = image.load_img(HOMEDIR + "tools/utils/target_img_ukb_space.nii.gz")

# Load MNI gm mask
gm_mask_raw = image.load_img(HOMEDIR + "tools/utils/mni_icbm152_gm_tal_nlin_asym_09c.nii")

# Binarize
gm_mask_binarized = image.math_img(f'img > {GM_THR}', img=gm_mask_raw)

# Resample
gm_mask = image.resample_img(
        gm_mask_binarized, target_affine=target_img.affine,
        target_shape=target_img.shape, interpolation="nearest",
        fill_value=0)

# %%
# =============================================================================
# Build regressor matrix
# =============================================================================

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
            }

    for name, type_ in var_dict.items():

        check_covariance(
                regressors_clean,
                var1=name,
                var2="age",
                type1=type_,
                type2="cont",
                save=True,
                prefix=OUTDIR + "covariance/pub_meta_neurofunction_covar"
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
                prefix=OUTDIR + "covariance/pub_meta_neurofunction_covar"
                )

        plt.close("all")

    if RLD == False:
        # Match
        regressors_matched = match(
                df=regressors_clean,
                main_var="diab",
                vars_to_match=["age", "sex", "college"],
                N=1,
                random_state=1
                )

# Status
print("Covariance checked.")

if RLD == False:
    # Save matched regressors matrix
    regressors_matched.to_csv(OUTDIR + f"regressors/pub_meta_neurofunction_matched_regressors_{CTRS}.csv")

    # Status
    print("Matching performed.")


# =============================================================================
# Analysis
# =============================================================================

# Prep
# ------
# Status
print("Loading regressors.")

# Load regressors
regressors_matched = pd \
    .read_csv(OUTDIR + \
         f"regressors/pub_meta_neurofunction_matched_regressors_{CTRS}.csv",
         index_col=0) \
    .rename({"eid": "subject_label"}, axis=1)


# Final sample sizes
print("Total sample size:", regressors_matched.shape[0])
print("Sample sizes:", regressors_matched.groupby([CTRS]).count()["subject_label"])

# Status
print("Loading nifti images.")

# List of input images, careful with the order!
imgs = [image.load_img(SRCDIR + f"{MDL}/{MDL.lower()}_normalized_batch_{BATCH}/" \
                       f"{sub}_{MDL.lower()}_normalized.nii") \
        for sub in regressors_matched["subject_label"].astype(int)]

# Status
print("Preparing Nistats design matrix.")

# Get subject labels
subject_labels = regressors_matched["subject_label"]

# Make nistats deisgn matrix object
design_matrix = make_second_level_design_matrix(
        subject_labels,
        regressors_matched[regressors_dict[CTRS]]
        )

# Run regression
# -----------

# Status
print("Fitting model.")

# Construct second level GLM object
GLM = SecondLevelModel(mask_img=gm_mask, verbose=2)

# Fit GLM
GLM.fit(imgs, design_matrix=design_matrix)

# Extract results that belong to contrast of interest,
# approximating t scores with z scores due to high N
z_map = GLM.compute_contrast(second_level_contrast=CTRS,
                             output_type="z_score")

# Save Z map
nib.save(z_map, OUTDIR + f"stats/pub_meta_neurofunction_zmap_{MDL.lower()}_" \
                 f"batch{BATCH}_GM_{GM_THR}_contrast_{CTRS}{EXTRA}.nii")


# Threshold voxel maps
# -----------

# Status
print("Thresholding z-map.")

# Apply statistical thresholds
thresholded_map, threshold = map_threshold(
    z_map, alpha=PTHR, height_control='fdr', cluster_threshold=CT)

# Change zeros to nans
thr_map_nan = image.math_img('np.where(img == 0, np.nan, img)', img=thresholded_map)

# Export thresholded image
nib.save(thr_map_nan, OUTDIR + f"stats/pub_meta_neurofunction_statthr_" \
         f"{MDL.lower()}_batch{BATCH}_GM_{GM_THR}_contrast_{CTRS}_uc{CT}_" \
         f"fdr{PTHR}{EXTRA}.nii")


# Status
print("Finished with extracting basic results.")

# %%
# =============================================================================
#  Check assumptions in maunally selected clusters
# =============================================================================

# Create labeled mask for each region
# ------

# Define clusters to look at
clusters_dict = {
        "Caudate": [11, 3, 16],
        "Orbitofrontal_Cortex": [8, 12, -18],
        "Premotor_Cortex": [-52, 10, 28],
        "Posterior_Cingulate_Gyrus": [-4, -34, 32]
            }

# Status
print("/n/n#####/nChecking assumptions around the following coordinates:")
pprint(clusters_dict)

# Status
print("Building masker object.")

# Function that return radius in number of voxels
comp_radius = lambda volume, voxel_dim: (volume*3/4/3.14159)**(1/3)/voxel_dim

# Transfrom cluster info
clusters = pd \
    .DataFrame(clusters_dict) \
    .T \
    .reset_index() \
    .set_axis(["label", "x", "y", "z"], axis=1) \
    .assign(**{
            "radius": comp_radius(VOLUME, VOXEL_DIM),
            "index": np.arange(len(clusters_dict))+1
            })

# Create clusters in 3d space based on center and radius
# assign labels to clusters
# and then merge them into one array sequentially

# Make voxels within the sphere n, others 0
def create_bin_sphere(arr_size=None, center=None, r=None, value=None):
    """
    Function from SO: https://stackoverflow.com/a/56060957/11411944
    Make 3d sphere
    """
    coords = np.ogrid[:arr_size[0], :arr_size[1], :arr_size[2]]
    distance = np.sqrt((coords[0] - center[0])**2 + \
                       (coords[1]-center[1])**2 + \
                       (coords[2]-center[2])**2)
    return value*(distance <= r)

# MNI to work space coordinate conversion
aff = target_img.affine
mni_convert = lambda mni_coords, aff: \
                    np.linalg.inv(aff[:3, :3])@(mni_coords - aff[:3, 3])

# Get all zero mask to serve as 3d space
space = np.array(target_img.get_data())
space[:] = 0

# Array which will be the mask with labels
label_mask = np.copy(space)

# Do for every cluster
for (i, cluster) in clusters.iterrows():

    # Construct 3d sphere
    current_cluster = create_bin_sphere(
            arr_size=space.shape,
            center=mni_convert(cluster[["x", "y", "z"]], aff),
            r=(cluster["radius"]-0.5),
            value=cluster["index"]
            )

    # Print size
    print(f'size of cluster #{int(cluster["index"])}:',
          np.argwhere(current_cluster == cluster["index"]).shape[0])

    # Add label cluster to mask, sequentially
    label_mask = \
        np.where(current_cluster == cluster["index"],
                 current_cluster,
                 label_mask)

# Conver label_mask to nifti
mask_img = image.new_img_like(
        target_img, data=label_mask
        )
# Save mask img
#nib.save(mask_img, "/Users/botond/Desktop/mask_img.nii")

# Construct masker object (with gm mask also added in)
label_masker = \
    input_data.NiftiLabelsMasker(mask_img, background_label=0, mask_img=gm_mask)


# Mask ALFF in subject data
# ------
# Assuming that the lsit named "imgs" was not manipulated and has images in the
# same order as "regressors_matched"

# Status
print("Applying masking.")

# Get subjects


# Dictionary to store data
signal_dict = {}

# Iterate through all subjects
for i, subject in tqdm(enumerate(regressors_matched["subject_label"]),
                       total=len(regressors_matched["subject_label"]),
                       desc="Extracting signal from subject: "):

    # Apply masking to ALFF image
    signal = label_masker.fit_transform([imgs[i]])

    # Construct dictionary
    signal_dict[subject] = list(signal[0])

# Status
print("Transforming extracted signal.")

# Transform into dataframe
data = pd \
    .DataFrame(signal_dict.items()) \
    .set_axis(["subject_label", "data"], axis=1) \
    .explode("data") \
    .assign(**{"label": \
               [item for item in clusters["label"].to_list()] \
               *len(regressors_matched["subject_label"])}) \
    .pivot(index="subject_label", columns="label", values="data") \
    .astype(float) \
    .reset_index()

# Build full matrix
df = regressors_matched \
    .merge(data, on="subject_label", how="inner") \
    .rename({"subject_label": "eid"}, axis=1)

# Run stat model
# -----

# Run stats for each label
for label in clusters["label"]:

    # Status
    print(f"Fitting model for {label}.")

    # Formula
    if CTRS == "age":
        formula = f"{label} ~ age + C(sex) + C(college)"
    if CTRS == "diab":
        formula = f"{label} ~ C(diab) + age + C(sex) + C(college)"

    # Construct OLS model
    model = smf.ols(formula, data=df)

    # Fit model
    results = model.fit()

    # Save results
    with open(OUTDIR + f"stats_misc/pub_meta_neurofunction_regression_table" \
              f"_{label}_{CTRS}.html", "w") as file:
        file.write(results.summary().as_html())

    # Check assumptions
    check_assumptions(
        results,
        df,
        prefix=OUTDIR + \
        f"stats_misc/pub_meta_neurofunction_stats_assumptions_{label}_{CTRS}"
        )

    if CTRS == "diab":

        # Status
        print(f"Visualizing relationships for {label}.")

        # Bin age
        bins = np.arange(0, 100, 5)
        df["age_group"] = pd.cut(df["age"], bins).astype(str)

        # Sort per age
        df = df.sort_values(by="age")

        # Plot for age*diabetes for current label
        plt.figure()
        sns.lineplot(data=df, x="age_group", y=f"{label}",
                   hue="diab", ci=68, palette=sns.color_palette(["black", "red"]),
                   err_style="bars", marker="o", sort=False,
                   err_kws={"capsize": 5}) \

        plt.savefig(OUTDIR + f"stats_misc/pub_meta_neurofunction_age-diab-plot_{label}.pdf")
        plt.close()

# Status
print("Finished execution.")
