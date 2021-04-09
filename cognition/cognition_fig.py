#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:04:45 2020

@author: botond

Notes:
This script generates a multi-panel figure from cognitive data.

"""

import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtc
import seaborn as sns
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers import plotting_style
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'cognition')
get_ipython().run_line_magic('matplotlib', 'inline')

# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/cognition/"

# Inputs
# Case specific values
cases = ["age", "diab", "meta"]
titles = [
        "Age: UK Biobank dataset\n(HC only, education " \
            "and sex-matched)",
        "T2DM: UK Biobank dataset\n(T2DM+ vs. HC, age, education " \
            "and sex-matched)",
        "T2DM: meta-analysis of published literature\n(T2DM+ vs. HC, age," \
        "education and sex-matched)",
        ]
ylabeltexts = [
        "Percentage change in task performance\nacross age (% per year)",
        "Percentage difference in task performance\nT2DM+ vs. HC (%)",
        "Standardized mean difference\nT2DM+ vs. HC (Cohen's d)"
        ]
colors = ["Purples", "Blues", "bone_r"]
ylims = [[-2.5, 0.3], [-17.0, 2.5], [-0.75, 0.2]]
sfs = [4e4, 1e3, 0.3]  # Marker size factors
sfscf = [15000, 2000, 25]  # Marker size scale factors
sdxo = [0.76, 0.75, 0.9]  # x axis offset of scale info
textpads = [0.1, 0.2, 0.02]  # Padding for text along y axis
xtickrots = [0, 0, 0]  # Rotation of xticks
xtickvas = ["top", "top", "top"]  # Vertical alignment for xticks
xtickpads = [0, 0, 0]  # Paddong fo xticks

#raise

# %%

# Load data
# ------

# Dict to store data
data = {}

# Iterate over all cases (age, diab, meta...)
for case in cases:

    # Load
    df = pd.read_csv(
            OUTDIR + f"stats/pub_meta_cognition_stats_{case}.csv",
            index_col=0
            )

    # Transfomrations specific to case
    if case in ["age", "diab"]:

            # Cast strings to float arrays
            df["sample_sizes"] = \
                df["sample_sizes"].apply(lambda item: np.array(item[1:-1].split(", "), dtype=float))
            df["conf_int"] = \
                df["conf_int"].apply(lambda item:
                    np.array([float(val) for val in item[1:-1].split(" ") if len(val) > 0])
                )

    elif case in ["meta"]:

        # Transform df
        df = df \
            .reset_index() \
            .pipe(lambda df:
                df.assign(**{
                    "label": df["Cognitive Domain"],
                    "conf_int": [[row[1]["LB"], row[1]["UB"]] \
                                 for row in df.iterrows()]
                            })
                )

    else:
        raise(ValueError("Unknown case!"))

    # Order
    if case == cases[0]:
        order = df.sort_values(by="beta").reset_index(drop=True)["label"]
        order_dict = dict((v,k) for k,v in order.to_dict().items())

    df = df.sort_values(by="label", key=lambda x: x.map(order_dict), ignore_index=True)

    # Assign transformed df to data dict
    data[case] = df


# %%
# =============================================================================
# Figure
# =============================================================================

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

# Figure
f = plt.figure(figsize=(8.5, 11))
plt.suptitle("Domain specific cognitive deficits associated with age and T2DM\n")

# Panels A & B
# ------

for c, case in enumerate(cases):

    # Current case's dataframe
    df = data[case]

    # Add new line character into x labels
    df["label"] = df["label"].str.replace("_", "\n")
    df["label"] = df["label"].str.replace(" ", "\n")
    df["label"] = df["label"].str.replace("Short\nTerm\nMemory", "Short-Term\nMemory")

    # Sort labels alphabetically
#    df = df.sort_values(by="label", ignore_index=True)

    # Pick subplot
    plt.subplot(len(cases), 1, c+1)

    # Populate plot
    if case in ["age", "diab"]:

        # Colors
        colors_all = colors_from_values(
            np.array(list(-df["beta"]) + [df["beta"].min() + 2, df["beta"].max()]),
            colors[c])

        for i, item in enumerate(df.iterrows()):

            # Extract data
            x = item[0]
            ss, p, y, t, conf_int  = \
                item[1][["sample_sizes", "pval", "beta", "tval", "conf_int"]]

            conf_dist = abs(y - np.array(conf_int))[:, None]

            # Blob for representing value and sample size
            plt.scatter(x=x, y=y, s=sum(ss)**2/sfs[c], color=colors_all[i])
                        #"mediumblue")

            # Plot center of estimate
            plt.scatter(x=x, y=y, s=15*lw, color="k")

            # Errorbars
            plt.errorbar(x, y, yerr=conf_dist, capsize=4*lw, capthick=0.75*lw,
                         elinewidth=0.75*lw, color="black")

        #    # Annotate stats as text
        #    text = f"T={t:.1f} \n {pformat(p)}" + p2star(p) \
        #        +  f"\n$\mathbf{{N_{{T2DM}}}}$={ss[1]:,}\n$\mathbf{{N_{{ctrl}}}}$={ss[0]:,}"
        #
        #    text_y = 0.2 # 0.5 if max(conf_int) < 0 else max(conf_int) + 0.5
        #    va = "bottom" # "bottom" if y > 0 else "top"
        #    plt.annotate(text, xy=[x, text_y],
        #                 fontsize=9*fs, ha="center", va=va)

            # Add statistical asterisks
            text = p2star(p)
            text_x = x + 0.00
            text_y =  min(conf_int) - textpads[c] \
                    if y < 0 else max(conf_int) + textpads[c]
            va = "top" if y < 0 else "bottom"
            plt.annotate(text, xy=[text_x, text_y], fontsize=12*fs,
                         ha="center", va=va, fontweight="bold",
                         rotation=0)

    elif case in ["meta"]:

        # Colors
        colors_all = colors_from_values(
            np.array(list(-df["EFFS"]) + [df["EFFS"].min() + 1, df["EFFS"].max()]),
            colors[c])[:-2]


        for i, item in enumerate(df.iterrows()):

            # Extract data
            x = item[0]
            y, conf_int, K, Q, I2, p  = \
                item[1][["EFFS", "conf_int", "K", "Q", "I2", "p"]]

            conf_dist = abs(y - np.array(conf_int))[:, None]

            # Blob for representing value and sample size
            plt.scatter(x=x, y=y, s=K**2/sfs[c], color=colors_all[i])

            # Plot center of estimate
            plt.scatter(x=x, y=y, s=15*lw, color="k")

            # Errorbars
            plt.errorbar(x, y, yerr=conf_dist, capsize=4*lw, capthick=0.75*lw,
                         elinewidth=0.75*lw, color="black")

        #    # Annotate stats as text
        #    text = \
        #        f"K={K}" \
        #        f"\nQ={Q}" \
        #        f"\n$\mathbf{{I^2}}$={I2}" \
        #        f"\n{pformat(p)}" + p2star(p)
        #

        #    f"\n {pformat(p)}" + p2star(p) \
        #        +  f"\n$\mathbf{{N^{2}$={ss[1]}\n$\mathbf{{N_{{ctrl}}}}$={ss[0]}"

        #    text_y = 0.2 # 0.5 if max(conf_int) < 0 else max(conf_int) + 0.5
        #    va = "bottom" # "bottom" if y > 0 else "top"
        #    plt.annotate(text, xy=[x, text_y],
        #                 fontsize=6.5*fs, ha="center", va=va)


            # Add statistical asterisks
            text = p2star(p)
            text_x = x + 0.00
            text_y =  min(conf_int) - textpads[c] \
                    if y < 0 else max(conf_int) + textpads[c]
            va = "top" if y < 0 else "bottom"
            plt.annotate(text, xy=[text_x, text_y], fontsize=8*fs,
                         ha="center", va=va, fontweight="bold",
                         rotation=0)

    # Format
    # Add title
    plt.title(titles[c])

    # Limits
    plt.xlim([-0.5, len(df)-0.5])
    plt.ylim(ylims[c])

    # Labels
    plt.ylabel(ylabeltexts[c])
    if c == len(cases)-1:
        plt.xlabel("\nCognitive domains")

    plt.gca().get_yaxis().set_major_formatter(
            mtc.FuncFormatter(lambda x, p: format(f"{x:.1f}")))

    # Ticks/lines
    plt.axhline(0, linewidth=0.5*lw, color="black", dashes=[4, 4])
    plt.xticks(ticks=np.arange(len(df)), labels=df["label"],
               rotation=xtickrots[c], va=xtickvas[c])
    plt.gca().tick_params(axis="x", pad=xtickpads[c])
    plt.gca().xaxis.tick_bottom()
    plt.gca().yaxis.tick_left()

    for sp in ['bottom', 'top', 'left', 'right']:
        plt.gca().spines[sp].set_linewidth(0.5*lw)
        plt.gca().spines[sp].set_color("black")

    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)

    # Add scale
    plt.scatter(x=len(df)-sdxo[c], y=ylims[c][0] * 0.88, s=sfscf[c]**2/sfs[c],
                color="gray")

    plt.annotate(text=f"Scale:\nN={sfscf[c]}",
                 xy=[len(df)-sdxo[c], ylims[c][0] * 0.88], va="center", ha="center")

## Caption info, !: not quite correct, cases with scores are more complex than
## just > 0. FOr some it's >=0, and others it's >0
## ---------
#
#stn = list(df.columns).index("f4282")
#id_vars = df.columns[:stn]
#
## Total sample size
#
## Gross total sample size and age
#print(
#  df \
#    .melt(id_vars=id_vars, var_name="feat", value_name="score") \
#    .query(f'feat not in {excl}') \
#    .dropna(subset=["score"], axis=0) \
#    .query('score > 0') \
#    .fillna(value={"age_onset": "NAN"}, axis=0) \
#    .pivot_table(
#            index=['eid', 'age', 'college', 'diab', 'age_onset'],
#            columns="feat") \
#    .reset_index() \
#    .groupby("diab") \
#    ["age"] \
#    .describe()
#    )
#
## Sample sizes and Age per cognitive task
#print(df \
#    .melt(id_vars=id_vars, var_name="feat", value_name="score") \
#    .query(f'feat not in {excl}') \
#    .dropna(subset=["score"], axis=0) \
#    .query('score > 0') \
#    .rename({"feat": "index"}, axis=1) \
#    .merge(labels, on="index") \
#    .groupby(["label", "diab"])["age"] \
#    .describe())
#
## Beta coefficients
#print(df[["label", "beta", "tval", "pval"]])

#print(df.groupby("diab")["age"].describe())

# Save
# ------
plt.tight_layout(h_pad=3)
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_cognition.pdf",
            transparent=True)

plt.close()
