from typing import List


class ExperimentResultsAE:
    epochs: List[int] = []
    train_metric: List = []
    test_metric: List = []

    def add_results(self, epoch_nr, metric, is_in_test_mode):
        if epoch_nr not in self.epochs:
            self.epochs.append(epoch_nr)

        if is_in_test_mode:
            self.test_metric.append(metric)
        else:
            self.train_metric.append(metric)
