from typing import List


class ExperimentResultsAE:
    def __init__(self, validation_mode):
        self.epochs: list[int] = []
        self.train_metric: list = []
        self.test_metric: list = []
        self.final_test_metric: float = -1
        self.using_validation_set = validation_mode != 'none'

    def add_results(self, epoch_nr, metric, is_in_test_mode):
        if epoch_nr not in self.epochs:
            self.epochs.append(epoch_nr)

        if is_in_test_mode:
            self.test_metric.append(metric)
        else:
            self.train_metric.append(metric)

    def to_json(self):
        return {
            "epochs": self.epochs,
            "train_metric": self.train_metric,
            "validation_metric" if self.using_validation_set else "test_metric" : self.test_metric,
            "final_test_metric": self.final_test_metric if self.using_validation_set else self.test_metric[-1]
        }