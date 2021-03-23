#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:49:07 2021

@author: botond
"""

import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtc
import seaborn as sns
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/med/cognition/"

# Inputs
CTRS = "metfonly_unmed"

# Case specific values
cases = [CTRS]
titles = [
        ""
        ]
ylabeltexts = [
        "Difference in Task Performance\n{cases[0]} (% of Avg)",
        ]
colors = ["Purples"]
ylims = [[-18, 10]]
sfs = [1e2]  # Marker size factors
textpads = [0.1]  # Padding for text along y axis
xtickrots = [0]  # Rotation of xticks
xtickvas = ["top"]  # Vertical alignment for xticks
xtickpads = [0]  # Paddong fo xticks

#raise

# %%
# Suppprt functions
# ------

# Function to convert float p values to str
def float_to_sig_digit_str(x, k):
    """
    Converts float to string with one significant figure
    while refraining from scientific notation

    inputs:
        x: input float to be converted to string (float)
        k: number of significant figures to keep (int)
    """

    import numpy as np

    # Get decimal exponent of input float
    exp = int(f"{x:e}".split("e")[1])

    # Get rid of all digits but the first figure
    x_fsf = round(x*10**-exp, k-1) * 10**exp

    # Get rid of scientific notation and convert to string
    x_str = np.format_float_positional(x_fsf)

    # Return string output
    return x_str

# Add p values and sample sizes to plot
def pformat(p):
    """ Formats p values for plotting """

#    if p < 0.001:
#        return "$\it{P}$=" + float_to_sig_digit_str(p, 1)
#    elif p > 0.995:
#        return "$\it{P}$=1.0"
#    else:
#        return "$\it{P}$=" + f"{p:.2g}"

    if p > 0.995:
        return "$\it{P}$=1.00"
    elif p >= 0.01:
        return "$\it{P}$=" + f"{p:.2f}" # [1:]
    elif p >= 0.001:
        return "$\it{P}$=" + f"{p:.3f}" # [1:]
    elif p < 0.001:
        return "$\it{P}$<0.001"
    else:
        "INVALID!"


def p2star(p):
    if p > 0.05:
        return ""
    elif p > 0.01:
        return "*"
    elif p > 0.001:
        return "**"
    else:
        return "***"

# Colors
def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)


# Styling
# -------

plt.style.use("default")
#plt.style.use("ggplot")
#sns.set_style("whitegrid")

fs=1.3  # Fontsize
lw=2  # Linewidth

# Stylesheet
plt.rcParams['xtick.color'] = "black"
plt.rcParams['ytick.color'] = "black"
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['figure.titlesize'] = 16*fs
plt.rcParams['figure.titleweight'] = "bold"
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['text.color'] = "black"
plt.rcParams['axes.labelcolor'] = "black"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12*fs
plt.rcParams['xtick.labelsize']=10.5*fs
plt.rcParams['ytick.labelsize']=11*fs
plt.rcParams['axes.labelsize']=11*fs
plt.rcParams['axes.labelweight'] = "bold"
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 20*fs
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rcParams['axes.titlesize'] = 13*fs
plt.rcParams['axes.titleweight'] = "bold"
#plt.rcParams['axes.axisbelow'] = True

# Load data
# ------

# Dict to store data
data = {}

# Iterate over all cases (age, diab, meta...)
for case in cases:

    # Load
    df = pd.read_csv(
            OUTDIR + f"stats/pub_meta_med_cognition_stats_{case}.csv",
            index_col=0
            )

    # Cast strings to float arrays
    df["sample_sizes"] = \
        df["sample_sizes"].apply(lambda item: np.array(item[1:-1].split(", "), dtype=float))
    df["conf_int"] = \
        df["conf_int"].apply(lambda item:
            np.array([float(val) for val in item[1:-1].split(" ") if len(val) > 0])
        )

    # Assign transformed df to data dict
    data[case] = df

# =============================================================================
# Figure
# =============================================================================

f = plt.figure(figsize=(19.2, 7))
plt.suptitle("Cognitive Performance as a Function of Metformin Medication Status:" \
          "\nUK Biobank Dataset, Coarse Matched for Age, Disease Duration and Sex\n")

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
    df = df.sort_values(by="label", ignore_index=True)

    # Pick subplot
    plt.subplot(len(cases), 1, c+1)

    # Populate plot
    # Colors
    colors_all = colors_from_values(
        np.array(list(-df["beta"]) + [df["beta"].min() + 4, df["beta"].max()]),
        colors[c])[:-2]

    for i, item in enumerate(df.iterrows()):

        # Extract data
        x = item[0]
        ss, p, y, t, conf_int  = \
            item[1][["sample_sizes", "pval", "beta", "tval", "conf_int"]]

        conf_dist = abs(y - np.array(conf_int))[:, None]

        # Blob for representing value and sample size
        plt.scatter(x=x, y=y, s=sum(ss)**2/sfs[c], color=colors_all[i])
                    #"mediumblue")

        # Errorbars
        plt.errorbar(x, y, yerr=conf_dist, capsize=12, capthick=lw,
                     elinewidth=lw, color="black")

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

    # Format
    # Add title
    plt.title(titles[c])

    # Limits
    plt.xlim([-0.5, len(df)-0.5])
    plt.ylim(ylims[c])

    # Labels
    plt.ylabel(ylabeltexts[c])
    if c == len(cases)-1:
        plt.xlabel("\nCognitive Domains")

    plt.gca().get_yaxis().set_major_formatter(
            mtc.FuncFormatter(lambda x, p: format(f"{x:.1f}")))

    # Ticks/lines
    plt.axhline(0, linewidth=lw, color="black", dashes=[4, 4])
    plt.xticks(ticks=np.arange(len(df)), labels=df["label"],
               rotation=xtickrots[c], va=xtickvas[c])
    plt.gca().tick_params(axis="x", pad=xtickpads[c])
    plt.gca().xaxis.tick_bottom()
    plt.gca().yaxis.tick_left()

    for sp in ['bottom', 'top', 'left', 'right']:
        plt.gca().spines[sp].set_linewidth(lw)
        plt.gca().spines[sp].set_color("black")

    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)

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
plt.tight_layout(h_pad=2)
plt.savefig(OUTDIR + f"figures/JAMA_meta_figure_med_cognition_{CTRS}.pdf",
            transparent=True)

plt.close()
