import numpy as np

def calc_avg(vec):
    n = len(vec)
    if n == 0:
        return 0
    return np.matmul(np.transpose(vec), np.ones((n))) / n

def determine_deviation_softmax(softmax, true_labels, class_to_idx_mapper):
    '''
    Calculates the positions of the true label in a softmax output.
    softmax : (N,M) array_like
        Matrix of softmax results.
    y : (N,) array_like
        Input array of the true labels.
    '''
    idxs = np.apply_along_axis(arr=true_labels, 
                               func1d=lambda x: class_to_idx_mapper(x),
                               axis=0)
    idxs = idxs.reshape((idxs.shape[0], 1))

    softmax_ranks = np.argsort(softmax, axis=1)[:, ::-1]
    positions = softmax_ranks == idxs
    idx_matrix = np.array([np.arange(positions.shape[1])] * positions.shape[0])
    return idx_matrix[positions]

def find_type(collection, type):
    for val in collection:
        if isinstance(val, type):
            return val
        
    return None

def random_splits(splits):
    idxs = []
    for i, split in enumerate(splits):
        idxs = idxs + [i] * split
    
    idxs = np.array(idxs)
    np.random.shuffle(idxs)

    return idxs

class LatexTableRow:
    def __init__(self,
                 lower_success_bound,
                 true_success,
                 mse,
                 avg_success,
                 avg_success_correct,
                 avg_success_incorrect,
                 avg_out_dev_incorrect,
                 pears_coef) -> None:
        self.lower_success_bound = lower_success_bound
        self.true_success = true_success
        self.mse = mse
        self.avg_success = avg_success
        self.avg_success_correct = avg_success_correct
        self.avg_success_incorrect = avg_success_incorrect
        self.avg_out_dev_incorrect = avg_out_dev_incorrect
        self.pears_coef = pears_coef

class LatexTableBuilder:
    def __init__(self) -> None:
        self.table = ""
        self.table += "\\begin{tabular}{lccccccccc} \\toprule \n"
        self.table += "\\multirow{2}{*}{Model} & \\multicolumn{3}{c}{Metric \\ref{item:m1} and \\ref{item:m2}} & \\multicolumn{3}{c}{Metric \\ref{item:m3}} & \\multicolumn{3}{c}{Metric \\ref{item:m4}} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \n"
        self.table += "& $\\lambda$ & $\\lambda^*$ & MSE & \\avgsucc & \\avgsucc (\\cmark) & \\avgsucc (\\xmark) & \\avgpos (\\cmark) & \\avgpos (\\xmark) & Pearson corr. \\\\ \\midrule \n"

    def add_mnist(self, model_to_rows):
        self._add_dataset(model_to_rows, {"Best": "$\\mnistbest$", "Avg": "$\\mnistavg$", "Worst": "$\\mnistworst$"})

    def add_cifar(self, model_to_rows):
        self._add_dataset(model_to_rows, {"Best": "$\\cifarbest$", "Avg": "$\\cifaravg$", "Worst": "$\\cifarworst$"})

    def build(self):
        self._replace_last_occurrence("\\hline\\hline \n", "\\bottomrule \\\\ \n")
        self.table += "\\multicolumn{10}{c}{*** With p-value  $< 0.0001$ indicating high statistical significance} \n"
        self.table += "\\end{tabular}"
        return self.table

    def _to_latex_row(self, row, model_name_latex, last_segment="\n"):
        return "{} & {} & {} & {} & {} & {} & {} & 0 & {} & {}(***) \\\\ {}".format(model_name_latex,
                                                                                  row.lower_success_bound,
                                                                                  row.true_success,
                                                                                  row.mse,
                                                                                  row.avg_success,
                                                                                  row.avg_success_correct,
                                                                                  row.avg_success_incorrect,
                                                                                  row.avg_out_dev_incorrect,
                                                                                  row.pears_coef,
                                                                                  last_segment)  
    
    def _add_dataset(self, model_to_rows, name_map):
        for level in ["Best", "Avg", "Worst"]:
            for model,row in model_to_rows.items():
                if model.name.endswith(level):
                    if level == "Best":
                        latex_row = self._to_latex_row(row, name_map[level])
                    elif level == "Avg":
                        latex_row = self._to_latex_row(row, name_map[level])
                    else:
                        latex_row = self._to_latex_row(row, name_map[level], last_segment="\\hline\\hline \n")
                    
                    self.table += latex_row

    def _replace_last_occurrence(self, old, new):
        last_index = self.table.rfind(old)  
        if last_index != -1: 
            self.table = self.table[:last_index] + new + self.table[last_index + len(old):]