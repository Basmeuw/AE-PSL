from typing import List


class ExperimentResults:
    def __init__(self):
        self.epochs: List[int] = []
        self.train_metric = []
        self.test_metric = []
        self.incoming_client_communication_overhead_in_mb: int = -1
        self.outgoing_client_communication_overhead_in_mb: int = -1
        self.params: dict = {}

    def add_results(self, epoch_nr, metric, is_in_test_mode):
        if epoch_nr not in self.epochs:
            self.epochs.append(epoch_nr)

        if is_in_test_mode:
            self.test_metric.append(metric)
        else:
            self.train_metric.append(metric)

    def set_client_communication_overhead(self, incoming_in_mb, outgoing_in_mb):
        """
        Sets the incoming and outgoing communication overhead for a client in mb, if not already set.
        Since the overhead is an average over all clients and the dataset length is constant, the values will be identical each epoch. Hence, the values are only set once.
        """
        if self.incoming_client_communication_overhead_in_mb != -1 or self.outgoing_client_communication_overhead_in_mb != -1:
            return

        self.incoming_client_communication_overhead_in_mb = incoming_in_mb
        self.outgoing_client_communication_overhead_in_mb = outgoing_in_mb

    # def get_aggregated_metric_per_epoch(self, metric: dict):
    #     aggregated_metric_per_epoch = []
    #     for epoch in self.epochs:
    #         epoch_metrics = [metric[client_nr][epoch] for client_nr in metric if epoch in metric[client_nr]]
    #         if epoch_metrics:
    #             aggregated_metric_per_epoch[epoch] = sum(epoch_metrics) / len(epoch_metrics)
    #     return aggregated_metric_per_epoch

    def to_json(self):
        return {
            "epochs": self.epochs,
            "train_metric": self.train_metric,
            "test_metric": self.test_metric,
            "incoming_client_communication_overhead_in_mb": self.incoming_client_communication_overhead_in_mb,
            "outgoing_client_communication_overhead_in_mb": self.outgoing_client_communication_overhead_in_mb,
            "job_params": self.params
        }
