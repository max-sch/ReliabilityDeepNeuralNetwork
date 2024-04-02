import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def histoplot(scores_correct, scores_incorrect, title, show_plot=False):
    n1 = len(scores_incorrect)
    n2 = len(scores_correct)
    n = n1 + n2 
    all_scores = np.ones((n))

    all_scores[:n1] = scores_incorrect
    types = ["Incorrect scores"] * n1

    all_scores[n1:] = scores_correct
    types = types + ["Correct scores"] * n2

    levels = np.unique(all_scores).tolist()
    ordered_levels = sorted(levels, reverse=True)
    ordered_levels = [str(v) for v in ordered_levels]

    all_scores_str = [str(v) for v in all_scores]
    d={"Success levels":all_scores_str, "Types":types}
    dataframe=pd.DataFrame(data=d)
    dataframe['Success levels'] = pd.Categorical(dataframe['Success levels'], ordered_levels)
    sns.histplot(dataframe, x="Success levels", hue="Types", discrete=True).set(title=title)
    #plt.xlabel('Categories')
    #plt.ylabel('Frequency')
    #plt.title('Histogram of Categorical Values')
    if show_plot:
        plt.show()

def scatterplot(scores, var_compare, title, title_var_compare, show_plot=False):
    d={"Success levels":scores, f"{title_var_compare}":var_compare}
    dataframe=pd.DataFrame(data=d)
    sns.scatterplot(dataframe, x="Success levels", y=f"{title_var_compare}").set(title=title)

    if show_plot:
        plt.show()
