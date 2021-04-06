#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:46:09 2021

@author: botond

if 91*109*91 then voxel size is 2mm (standard MNI space)

"""

# =============================================================================
# Setup
# =============================================================================

import os
from functools import reduce
from nilearn import plotting, image, datasets, masking
import nibabel as nib
from nistats.thresholding import map_threshold
from matplotlib import pyplot as plt
import matplotlib as mpl
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'neurofunction')
get_ipython().run_line_magic('matplotlib', 'auto')

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/neurofunction/"

# Inputs
MDL = "ALFF"
PTHR = 0.05
UC = 12
GM_THR = 0.5  # GM threshold
cut_coords = (12, 8, 16)

#raise

# Helper functions
# -----
# Look at extremes
def extremes(imgs):
    for img in imgs:
      print(img.get_fdata().min())
      print(img.get_fdata().max())

# Increase resolution
new_affine = lambda img, zoom: \
    nib.affines.rescale_affine(
            img.affine,
            img.shape,
            zoom,
            list(map(lambda x: x*zoom, img.shape))
            )

resample = lambda img, zoom: \
    image.resample_img(
            img,
            target_affine=new_affine(img, zoom),
            target_shape=
                list(map(lambda x: x*zoom, img.shape))
                )

# Load gray matter mask
# ---------
# Load mask
gm_img = image.load_img(HOMEDIR + "data/atlas/" \
                        "mni_icbm152_gm_tal_nlin_asym_09c.nii")

# MNI props
mni = datasets.load_mni152_template()

# Resample gm_mask
gm_img_rs = image.resample_img(
            gm_img,
            target_affine=mni.affine,
            target_shape=mni.shape,
            interpolation="nearest"
                )

# Binarize gm mask
gm_mask = image.math_img(f'img > {GM_THR}', img=gm_img_rs)

# Plotting prep
# -----
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

#raise

# %%
# =============================================================================
# UKB
# =============================================================================

suffix1 = f"batch7_GM_{GM_THR}_contrast_age"
suffix2 = f"batch7_GM_{GM_THR}_contrast_diab"
cmap = 'PuOr_r'
vm = 5

# Load images
back = image.load_img(OUTDIR \
               + f'stats/pub_meta_neurofunction_zmap_{MDL}_{suffix1}.nii')
fore = image.load_img(OUTDIR \
               + f'stats/pub_meta_neurofunction_zmap_{MDL}_{suffix2}.nii')

# (ALready masked and in MNI space)

# Threshold fore image
fore, _ = map_threshold(
   fore, alpha=PTHR, height_control='fdr', cluster_threshold=UC)


#back1 = resample(back1, 4)
#fore1 = resample(fore1, 4)

# Separate pos/neg values
fore_view = fore.get_fdata()
fore_view[fore_view != fore_view]=0
fore_pos = image.math_img('img > 0', img=fore)
fore_neg = image.math_img('img < 0', img=fore)

# Display
fig = plt.figure(figsize=(19.2, 8))
plt.suptitle("Overlaid effects in ALFF with respect to age and T2DM: " \
             "UK Biobank dataset")
display = plotting.plot_roi(back, cmap=cmap, cut_coords=cut_coords,
                            vmin=-vm, vmax=vm, black_bg=True, figure=fig,
                            colorbar=True, draw_cross=False)
display.add_contours(fore_pos, colors="red", alpha=1, linewidths=0.5)
display.add_contours(fore_neg, colors="blue", alpha=1, linewidths=0.5)

plt.savefig(OUTDIR + f'figures/JAMA_meta_neurofunction_{MDL}_age-diab_overlap.pdf',
            transparent=True)

#plotting.view_img(fore).open_in_browser()

# %%
# =============================================================================
# Neuroquery
# =============================================================================

# Inputs
suffix1 = "contrast_age"
suffix2 = "contrast_diab"

#cmap = 'PRGn'
vm = 3

# Load images
back = nib.load(HOMEDIR + "results/neuroquery/stats/pub_meta_neuroquery_" \
                   f"zmap_{suffix1}.nii")
fore = nib.load(HOMEDIR + "results/neuroquery/stats/pub_meta_neuroquery_" \
                   f"zmap_{suffix2}.nii")

# Resample
back = image.resample_img(back, target_affine=mni.affine, target_shape=mni.shape)
fore = image.resample_img(fore, target_affine=mni.affine, target_shape=mni.shape)

# Mask
back = image.threshold_img(back, threshold="0%", mask_img=gm_mask)
fore = image.threshold_img(fore, threshold="0%", mask_img=gm_mask)

# Inpsect
extremes([back, fore])
#prod = reduce(lambda a, b: a*b, back.shape)
#
#UC=12
#UC = UC/((91*10991)/prod)

# Threshold fore image
fore, _ = map_threshold(
   fore, alpha=PTHR, height_control='fdr', cluster_threshold=UC)

# Save thresholded for map
t = fore.get_fdata()


nib.save(fore, HOMEDIR + f"results/neuroquery/stats/pub_meta_neuroquery_statthr_" \
         f"GM_{GM_THR}_{suffix2}_uc{UC}_fdr{PTHR}.nii")

# Separate pos/neg values
fore_view = fore.get_fdata()
fore_view[fore_view != fore_view]=0
fore_pos = image.math_img('img > 0', img=fore)
fore_neg = image.math_img('img < 0', img=fore)

# Display
fig = plt.figure(figsize=(19.2, 8))
plt.suptitle("Overlaid effects in ALFF with respect to age and T2DM: " \
             "Meta-analysis from Neuroquery")
display = plotting.plot_roi(back, cmap=cmap, cut_coords=cut_coords,
                            vmin=-vm, vmax=vm, black_bg=True, figure=fig,
                            colorbar=True, draw_cross=False)
display.add_contours(fore_pos, colors="red", alpha=1, linewidths=1)
display.add_contours(fore_neg, colors="blue", alpha=1, linewidths=1)

plt.savefig(HOMEDIR + "results/neuroquery/figures/JAMA_meta_neuroquery_age" \
            "-diab_overlap.pdf", transparent=True)

plt.close("all")

# %%
# =============================================================================
# Colorbar
# =============================================================================
#plt.figure(facecolor="black")
#cbar = plt.colorbar(mpl.cm.ScalarMappable(
#        norm=mpl.colors.Normalize(vmin=-vm, vmax=vm),
#        cmap=plt.get_cmap(cmap)),
#        ticks=[-vm, -vm/2, 0, vm/2, vm])
#cbar.ax.set_yticklabels([-vm, -vm/2, 0, vm/2, vm],
#                        fontsize=12,
#                        weight="bold",
#                        color="white")
#
#cbar.ax.set_title("z-score",
#                  fontsize=12,
#                  weight="bold",
#                  color="white")
