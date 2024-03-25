import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def histoplot(scores_correct, scores_incorrect, title):
    all_scores = np.array(scores_correct + scores_incorrect)

    n1 = len(scores_incorrect)
    all_scores[range(n1)] = scores_incorrect
    types = ["Incorrect scores"] * n1

    n2 = len(scores_correct)
    all_scores[range(n1,n1+n2)] = scores_correct
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
    plt.show()
