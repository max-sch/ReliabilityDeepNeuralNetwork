import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def histoplot(scores_correct, scores_incorrect, title, show_plot=False):
    data_frame = _as_data_frame(scores_correct, scores_incorrect)
    sns.histplot(data_frame, x="Success levels", hue="Types", discrete=True).set(title=title)

    if show_plot:
        plt.show()

def boxplot(scores_correct, scores_incorrect, title, show_plot=False):
    data_frame = _as_data_frame(scores_correct, scores_incorrect)
    sns.boxplot(data=data_frame, x="Model", y="Success levels", hue="Types").set(title=title)

    if show_plot:
        plt.show()

def scatterplot(scores, var_compare, title, title_var_compare, show_plot=False):
    d={"Success levels":scores, f"{title_var_compare}":var_compare}
    dataframe=pd.DataFrame(data=d)
    sns.scatterplot(dataframe, x="Success levels", y=f"{title_var_compare}").set(title=title)

    if show_plot:
        plt.show()

def lineplot(num_runs, scores, show_plot=True):
    d={"Lower reliability bounds":scores, "Num runs": num_runs}
    sns.lineplot(pd.DataFrame(data=d), x="Num runs", y="Lower reliability bounds")

    if show_plot:
        plt.show()


def barplot(avg_scores, positions, num_samples, show_plot=False):
    d={"Avg success levels":avg_scores, "Softmax position":positions, "Num samples":num_samples}
    dataframe=pd.DataFrame(data=d)
    ax = sns.barplot(dataframe, x="Softmax position", y="Avg success levels", hue="Softmax position", legend=None)
    for i, v in enumerate(avg_scores):
        samples = num_samples[i]
        if samples != 0:
            ax.text(i, v + 0.01, str(samples), ha='center', color='black')

    if show_plot:
        plt.show()

def _as_data_frame(scores_correct, scores_incorrect, model_name="Model"):
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
    model = [model_name] * n
    d={"Success levels":all_scores_str, "Types":types, "Model":model}
    dataframe=pd.DataFrame(data=d)
    dataframe['Success levels'] = pd.Categorical(dataframe['Success levels'], ordered_levels)
    
    return dataframe
