from typing import List


class ExperimentResults:
    def __init__(self, validation_mode):
        self.epochs: List[int] = []
        self.train_metric = []
        self.test_metric = []
        self.final_test_metric: float = -1
        self.incoming_client_communication_overhead_in_mb: int = -1
        self.outgoing_client_communication_overhead_in_mb: int = -1
        self.params: dict = {}
        self.using_validation_set = validation_mode != 'none'

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


    def to_json(self):
        # if we don't use a validation set, we simply pick the last test score as final test metric
        return {
            "epochs": self.epochs,
            "train_metric": self.train_metric,
            "validation_metric" if self.using_validation_set else "test_metric" : self.test_metric,
            "final_test_metric": self.final_test_metric if self.using_validation_set else self.test_metric[-1],
            "incoming_client_communication_overhead_in_mb": self.incoming_client_communication_overhead_in_mb,
            "outgoing_client_communication_overhead_in_mb": self.outgoing_client_communication_overhead_in_mb,
            "job_params": self.params
        }
