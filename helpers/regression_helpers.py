#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:58:02 2021

@author: botond
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from tqdm import tqdm
from IPython import get_ipython
from matplotlib.backends.backend_pdf import PdfPages

get_ipython().run_line_magic('matplotlib', 'inline')



def check_covariance(df, var1, var2, type1, type2, save=False, prefix=None):
    """ Function to quantify interaction between two variables """

    # Check types
    types = ["cont", "disc"]
    check_type = lambda type_: 0 if type_ not in types else 1
    list(map(check_type, [type1, type2]))

    # Refine data
    tdf = df[["eid", var1, var2]]

    # Both continuous
    if (type1 == "cont") & (type2 == "cont"):

        # Statistics
        corr = stats.pearsonr(tdf[var1], tdf[var2])
        text = f'Covariance: \n{var1} vs {var2}\n####\n' \
               f"Pearson's r: r={corr[0]:.3f}, p={corr[1]:.2e}"
        print(text)

        # Visualize
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"{var1} vs {var2}")
        plt.scatter(tdf[var1], tdf[var2], s=1)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.subplot(1, 2, 2)
        plt.annotate(text, xy=[0.3, 0.65], xycoords="axes fraction")
        plt.axis('off')

    # If discrete and continous
    if (type1 == "disc") & (type2 == "cont"):

        # Extract
        gs = tdf.groupby(var1).groups
        g1 = tdf.loc[gs[0], :][var2]
        g2 = tdf.loc[gs[1], :][var2]

        # Statistics
        ttest = stats.ttest_ind(g1, g2)
        bicorr = stats.pointbiserialr(tdf[var1], tdf[var2])
        text = f'Covariance: \n{var1} vs {var2}\n####\n' \
               f'{var1}=0 mean: {g1.mean():.2f}\n{var1}=1 mean: {g2.mean():.2f}\n' \
               f't-test: T={ttest[0]:.2f}, p={ttest[1]:.2e}\n' \
               f"Point biserial r: r={bicorr.correlation:.3f}, p={bicorr.pvalue:.2e}"
        print(text)

        # Visualize
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(g1, alpha=0.7, bins=30, label=f"{var1}=0", density=True)
        plt.hist(g2, alpha=0.7, bins=30, label=f"{var1}=1", density=True)
        plt.xlabel(var2)
        plt.ylabel("density")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.annotate(text, xy=[0.3, 0.65], xycoords="axes fraction")
        plt.axis('off')

    # If discrete + discrete
    if (type1 == "disc") & (type2 == "disc"):

        # Statistics
        ctab = tdf.groupby([var1, var2]).count()["eid"].unstack(fill_value=0)
        chitest = stats.chi2_contingency(ctab)
        text = \
            f'Covariance: \n{var1} vs {var2}\n####\n' + \
            str(ctab) + "\n\n" \
            f"Chi2: {chitest[0]:2f}\n" \
            f"Pval: {chitest[1]:2e}\n\n" \
            f"balanced table: \n{pd.DataFrame(chitest[3])}"
        print(text)

        # Visualize
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        sns.barplot(
                data=ctab.stack().rename("count").reset_index(),
                x=var1, y="count", hue=var2
                )
        plt.subplot(1, 2, 2)
        plt.annotate(text, xy=[0.3, 0.65], xycoords="axes fraction")
        plt.axis('off')


    # If cont + disc -> #TODO
    if (type1 == "cont") & (type2 == "disc"):
        # Extract
        gs = tdf.groupby(var2).groups
        g1 = tdf.loc[gs[0], :][var1]
        g2 = tdf.loc[gs[1], :][var1]

        # Statistics
        ttest = stats.ttest_ind(g1, g2)
        bicorr = stats.pointbiserialr(tdf[var2], tdf[var1])
        text = f'Covariance: \n{var2} vs {var1}\n####\n' \
               f'{var2}=0 mean: {g1.mean():.2f}\n{var2}=1 mean: {g2.mean():.2f}\n' \
               f't-test: T={ttest[0]:.2f}, p={ttest[1]:.2e}\n' \
               f"Point biserial r: r={bicorr.correlation:.3f}, p={bicorr.pvalue:.2e}"
        print(text)

        # Visualize
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(g1, alpha=0.7, bins=30, label=f"{var2}=0", density=True)
        plt.hist(g2, alpha=0.7, bins=30, label=f"{var2}=1", density=True)
        plt.xlabel(var1)
        plt.ylabel("density")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.annotate(text, xy=[0.3, 0.65], xycoords="axes fraction")
        plt.axis('off')

    # Add in some space to the console
    print("\n\n")

    # Save figure
    if save:
        plt.tight_layout()
        plt.savefig(prefix + f"_{var1}_{var2}.pdf")
#    plt.close("all")


def check_assumptions(results, sdf, prefix):
    """ Function to check for the assumptions of linear regression """

    # Unpack residuals
    residuals = results.resid

    # Create a PdfPages object
    pdf = PdfPages(prefix + f".pdf")

    # Scatterplotting function
    def scdensplot(x, y, bins=20):
        sns.scatterplot(x=x, y=y, s=5, color=".15")
        sns.histplot(x=x, y=y, bins=bins, thresh=None, cmap="mako", cbar=True)
    #    sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)

    # Residuals distribution (heteroscedasticity, linearity)
    fig = plt.figure(figsize=(10, 7))
    plt.title("Distribution of residuals")
    sns.histplot(residuals, stat="density", bins=50)
    x = np.linspace(residuals.min(), residuals.max(), 1000)
    gaussian = stats.t.pdf(x, results.df_resid, *stats.t.fit(residuals)[1:])
    plt.plot(x, gaussian, color="red", lw=2)
    plt.xlabel("Residual")
    plt.xlim(residuals.mean()-residuals.std()*4,
             residuals.mean()+residuals.std()*4)
    plt.axvline(0, color="k")
    plt.tight_layout(w_pad=1)
    plt.close("all")
    pdf.savefig(fig, transparent=True)

    # Residuals qq plot (heteroscedasticity, linearity)
    fig = plt.figure(figsize=(10, 7))
    sm.qqplot(residuals, stats.t, line="s", distargs=(results.df_resid,),
              ax=plt.gca())
#    plt.axhline(0, color="k", dashes=[2, 2])
    plt.axvline(0, color="k", dashes=[2, 2])
    plt.title("QQ plot of residuals")
    plt.xlabel("Theoretical T")
    plt.ylabel("Residual")
    plt.tight_layout(w_pad=1)
    plt.close("all")
    pdf.savefig(fig, transparent=True)

    # Residuals normality test
#    print(stats.normaltest(residuals))

    # Residuals vs subject ID (autocorrelation)
    fig = plt.figure(figsize=(10, 7))
    plt.title("Residuals vs samples in order")
    plt.xlabel("Sample")
    plt.ylabel("Residual")
    temp = sdf["eid"].to_frame().assign(**{"residual": residuals}) \
        .sort_values(by="eid").reset_index(drop=True).reset_index()
    scdensplot(temp["index"], temp["residual"])
    plt.axhline(0, color="white", dashes=[2, 2])
    plt.tight_layout(w_pad=1)
    plt.close("all")
    pdf.savefig(fig, transparent=True)

    # Residuals vs fitted value (heteroscedasticity)
    fig = plt.figure(figsize=(10, 7))
    plt.title("Residuals vs fitted values")
    plt.xlabel("Fitted value")
    plt.ylabel("Residual")
    scdensplot(results.fittedvalues, residuals, bins=20)
    plt.axhline(0, color="white", dashes=[2, 2])
    plt.tight_layout(w_pad=1)
    plt.close("all")
    pdf.savefig(fig, transparent=True)

    # Close pdf
    pdf.close()

    ## Residuals vs real value
    #plt.figure()
    #plt.title("Residuals vs real values")
    #scdensplot(sdf["feat"], residuals)  #TODO: Is this a problem?
    #scdensplot(sdf["feat"], sdf["feat"] - sdf["feat"].mean())
    #
    ## Residuals vs variable (linearity)
    #print(stats.pearsonr(rdf["age"], residuals))
    #scdensplot(sdf["age"], residuals)
    #
    #scdensplot(sdf["age"], sdf["feat"])
    #scdensplot(sdf["age"], results.fittedvalues)


def match(df=None, main_var=None, vars_to_match=[], N=1, random_state=1):
    """
    Function to perform matching across chosen independent variables.
    Method: exact matching across specified covariates.
    """

    # Separate items per main variable
    exp_subs = df.query(f"{main_var} == 1")
    ctrl_subs = df.query(f"{main_var} == 0")

    # List of matched items, later to serve as a df
    mdf_list = []

    # List to store number of available subject
    candidate_numbers_list = []

    # Iterate over all subjects positive to the treatment
    for i, exp_sub in tqdm(enumerate(exp_subs.iterrows()),
                           total=exp_subs.shape[0],
                           desc="Matching subject: "):

        # Find control subjects that match along variables
        query_statement = " & ".join([f'{var} == {exp_sub[1][var]}' \
                                      for var in vars_to_match])

        candidates = ctrl_subs.query(query_statement)

        # Store numbers
        candidate_numbers_list.append(len(candidates))

        # If there is at least 1 match
        if candidates.shape[0] > 0:

            # Pick from candidates randomly
            picked_ctrl_subs = candidates.sample(n=N, random_state=random_state)

            # If found: Take out subject from ctrl_subs hat
            ctrl_subs = ctrl_subs \
                .merge(
                    picked_ctrl_subs,
                    on=ctrl_subs.columns.to_list(),
                    how="left",
                    indicator=True
                    ) \
                .query('_merge != "both"').drop("_merge", axis=1)

            # If found: add both subjects to mdf
            mdf_list.append(exp_sub[1].to_frame().T)
            mdf_list.append(picked_ctrl_subs)

        else:
            print(f'\nNo match found for: {int(exp_sub[1]["eid"])}')

    # Concat into df
    mdf = pd.concat(mdf_list, axis=0)

    # Analyze candidate availability
    candidate_numbers = pd.DataFrame(candidate_numbers_list, columns=["count"])
    print(f"Matching info:\nmatched subjects: N={mdf.shape[0]}\n", \
            "candidates:\n", candidate_numbers.describe())

    return mdf

# Multi-matching function
def multi_match(df=None, main_var=None, vars_to_match=[], N=1, random_state=1):
    """
    Function to perform matching across chosen independent variables.
    Method: exact matching across specified covariates.
    Considers multiple (2+) groups across main_var.
    But groups have to be complete.
    Practically speaking: good for matching across a variable with multiple
    discrete values (2+), but not ideal for continuous variables with many values.
    """


    # Separate items per main variable
    groups = df.groupby(main_var)
    groups_dir = {key: df.loc[indexes] for key, indexes in groups.groups.items()}
    keys = list(groups.groups.keys())

    # List of matched items, later to serve as a df
    mdf_list = []

    # List to store number of available subjects
    candidate_numbers_list = []

    # Iterate over all subjects in reference group
    for i, ref_sub in tqdm(enumerate(groups_dir[keys[0]].iterrows()),
                           total=groups_dir[keys[0]].shape[0],
                           desc="Matching subject: "):

        # Find control subjects that match along variables
        query_statement = " & ".join([f'{var} == {ref_sub[1][var]}' \
                                      for var in vars_to_match])

        # Start candidates dictionary
        candidates = {}

        # Iterate over all groups outside of reference
        for i, (key, subs) in enumerate(groups_dir.items()):
            # Skip reference group
            if i == 0:
                continue

            # Query and add to candidates
            candidates[key] = subs.query(query_statement)

        # Store candidate numbers
        min_num = min(len(val) for _, val in candidates.items())
        candidate_numbers_list.append(min_num)

        # If there is at least 1 match
        if min_num > 0:

            # If found: add ref subject to mdf
            mdf_list.append(ref_sub[1].to_frame().T)

            # Iterate over all groups outside of reference
            for i, (key, subs) in enumerate(candidates.items()):

                # Pick from candidates randomly
                picked_candidate = subs.sample(n=N, random_state=random_state)

                # If found: Take out subject from ctrl_subs hat
                groups_dir[key] = groups_dir[key] \
                    .merge(
                        picked_candidate,
                        on=groups_dir[key].columns.to_list(),
                        how="left",
                        indicator=True
                        ) \
                    .query('_merge != "both"').drop("_merge", axis=1)

                # If found: add subject to mdf
                mdf_list.append(picked_candidate)

        else:
            print(f'\nNo match found for: {int(ref_sub[1]["eid"])}')

    # Concat into df
    mdf = pd.concat(mdf_list, axis=0).convert_dtypes()

    # Analyze candidate availability
    candidate_numbers = pd.DataFrame(candidate_numbers_list, columns=["count"])
    print(f"Matching info:\nmatched subjects: N={mdf.shape[0]}\n", \
            "candidates:\n", candidate_numbers.describe())

    return mdf