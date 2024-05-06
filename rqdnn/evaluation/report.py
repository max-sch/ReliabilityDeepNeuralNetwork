from commons.ops import LatexTableRow, LatexTableBuilder
from os import mkdir
from os.path import join, exists

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class EvaluationReport:
    def __init__(self, model) -> None:
        self.model = model
        self.model_reports = {}

    def add(self, repetition, model_report):
        self.model_reports[repetition] = model_report

    def export(mnist_reports, cifar_reports, result_dir="./results"):
        EvaluationReport._export_to_table(mnist_reports, cifar_reports, result_dir)
        EvaluationReport._export_figures(mnist_reports, cifar_reports, result_dir)

    def _export_figures(mnist_reports, cifar_reports, result_dir):
        if not exists(result_dir):
            mkdir(result_dir)
        
        # Line plots of convergence behavior
        EvaluationReport._export_lineplots(mnist_reports, filename=join(result_dir, "mnistconv.png"))
        EvaluationReport._export_lineplots(cifar_reports, filename=join(result_dir, "cifarconv.png"))

        # Bar plots of output deviation
        EvaluationReport._export_barplots(mnist_reports, result_dir)
        EvaluationReport._export_barplots(cifar_reports, result_dir)

    def _export_lineplots(reports, filename):
        scores = []
        num_runs = []
        models = []
        for eval_report in reports:
            num_repetitions = len(eval_report.model_reports)
            for repetition in range(num_repetitions):
                model_report = eval_report.model_reports[repetition]
                
                success_probs = model_report.conv_behavior
                scores.extend(success_probs)
                num_runs.extend([*range(len(success_probs))])
                models.extend([model_report.model.name] * len(success_probs))
        
        plt.figure()

        d={"Lower reliability bounds":scores, "Number of runs":num_runs, "Models":models}
        plot = sns.lineplot(pd.DataFrame(data=d), x="Number of runs", y="Lower reliability bounds", hue="Models")
        
        plt.savefig(filename) 

    def _export_barplots(reports, result_dir):
        for eval_report in reports:
            avg_scores = []
            num_samples = []
            positions = []
            n = eval_report.model_reports[0].get("Softmax position to reliability correlation").num_pos
            avg_num_samples = np.zeros((n))
            avg_avg_scores = np.zeros((n))

            num_repetitions = len(eval_report.model_reports)
            for repetition in range(num_repetitions):
                model_report = eval_report.model_reports[repetition]
                metric = model_report.get("Softmax position to reliability correlation")
                
                for pos in range(metric.num_pos):
                    samples = int(metric.num_samples_per_pos[pos])
                    avg_score = metric.avg_scores[pos]

                    avg_num_samples[pos] += samples
                    avg_avg_scores[pos] += avg_score
                    avg_scores.append(avg_score)
                    num_samples.append(samples)
                    positions.append(pos)

            num_pos = len(avg_num_samples)
            avg_num_samples = avg_num_samples / num_repetitions 
            avg_avg_scores = avg_avg_scores / num_repetitions    
            pos_to_include = np.arange(num_pos)[avg_num_samples > 50]
            
            for i,pos in enumerate(positions):
                if pos not in pos_to_include:
                    avg_scores[i] = 0
                    num_samples[i] = 0
        
            plt.figure()

            d={"Avg success levels":avg_scores, "Softmax position":positions}
            plot = sns.barplot(d, x="Softmax position", y="Avg success levels", hue="Softmax position", legend=None)
            for pos in range(num_pos):
                if pos in pos_to_include:
                    avg_sample = avg_num_samples[pos]
                    avg_avg_score = avg_avg_scores[pos]
                    plot.text(pos, avg_avg_score + 0.02, str(avg_sample), ha='center', color='black')

            filename = join(result_dir, "barplot_{}.png".format(eval_report.model.name))
            plt.savefig(filename) 

    def _export_to_table(mnist_reports, cifar_reports, result_dir):
        builder = LatexTableBuilder()
        
        model_to_rows = {report.model:EvaluationReport._to_table_row(report) for report in mnist_reports}
        builder.add_mnist(model_to_rows)

        model_to_rows = {report.model:EvaluationReport._to_table_row(report) for report in cifar_reports}
        builder.add_cifar(model_to_rows)
        
        table = builder.build()

        if not exists(result_dir):
            mkdir(result_dir)
        latex_file = open(join(result_dir, "resulttable.tex"), 'w')
        latex_file.write(table)
        latex_file.close()

    def _to_table_row(eval_report):
        lower_success_bound = 0
        true_success = 0
        avg_success = 0
        avg_success_correct = 0
        avg_success_incorrect = 0
        avg_out_dev_incorrect = 0
        pears_coef = 0
        
        num_repititions = len(eval_report.model_reports)
        for repetition in range(num_repititions):
            model_report = eval_report.model_reports[repetition]

            lower_success_bound += model_report.conv_behavior[-1]
            
            true_success += model_report.get("True success probability").true_success
            
            avg_rel_metric = model_report.get("Average reliability score")
            avg_success += avg_rel_metric.avg_score 
            avg_success_correct += avg_rel_metric.avg_score_correct
            avg_success_incorrect += avg_rel_metric.avg_score_incorrect

            avg_out_dev_incorrect += model_report.get("Average output deviation").avg_output_deviation

            pears_coef += model_report.get("Pearson correlation").pears_coef.correlation

        lower_success_bound = round(lower_success_bound / num_repititions, 4)
        true_success = round(true_success / num_repititions, 4)
        avg_success = round(avg_success / num_repititions, 4)
        avg_success_correct = round(avg_success_correct / num_repititions, 4)
        avg_success_incorrect = round(avg_success_incorrect / num_repititions, 4)
        avg_out_dev_incorrect = round(avg_out_dev_incorrect / num_repititions, 1)
        pears_coef = round(pears_coef / num_repititions, 4)

        return LatexTableRow(lower_success_bound=lower_success_bound,
                             true_success=true_success,
                             mse="TBD",
                             avg_success=avg_success,
                             avg_success_correct=avg_success_correct,
                             avg_success_incorrect=avg_success_incorrect,
                             avg_out_dev_incorrect=avg_out_dev_incorrect,
                             pears_coef=pears_coef)

class ModelReport:
    def __init__(self, model) -> None:
        self.model = model
        self.results = {}
        self.conv_behavior = []

    def add_convergence(self, success_probs):
        self.conv_behavior = success_probs

    def add(self, metric):
        self.results[metric.name] = metric
    
    def get(self, metric_name):
        return self.results[metric_name]