import time

import numpy as np
import torch
from tqdm import tqdm

from trainers.implementations.classification.centralized_trainer import CentralizedTrainer
from trainers.implementations.experiment_results import ExperimentResults

from models.meta_transformer.base.data2seq import InputModality


class Profiler:
    """Helper class to manage CUDA timing events."""

    def __init__(self):
        self.stats = {
            "data_loading": [],
            "forward": [],
            "backward": [],
            "optimizer": [],
            "total_batch": []
        }
        self.current_events = {}

    def start(self, name):
        if torch.cuda.is_available():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.current_events[name] = event
        else:
            self.current_events[name] = time.time()

    def end(self, name):
        if torch.cuda.is_available():
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()  # Wait for GPU to finish this step
            start_event = self.current_events.pop(name)
            elapsed = start_event.elapsed_time(end_event)  # Returns milliseconds
        else:
            start_time = self.current_events.pop(name)
            elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds

        self.stats[name].append(elapsed)

    def print_summary(self):
        print("\n" + "=" * 50)
        print("PERFORMANCE PROFILING SUMMARY (ms per batch)")
        print("=" * 50)
        headers = ["Phase", "Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)", "% of Total"]
        print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10}")
        print("-" * 70)

        total_mean = np.mean(self.stats['total_batch']) if self.stats['total_batch'] else 1.0

        for key, values in self.stats.items():
            if not values: continue
            mean_v = np.mean(values)
            std_v = np.std(values)
            min_v = np.min(values)
            max_v = np.max(values)
            ratio = (mean_v / total_mean) * 100 if key != 'total_batch' else 100.0

            print(f"{key:<15} {mean_v:<10.2f} {std_v:<10.2f} {min_v:<10.2f} {max_v:<10.2f} {ratio:<10.1f}%")
        print("=" * 50 + "\n")


class ProfiledTrainer(CentralizedTrainer):
    """
    A subclass of CentralizedTrainer that injects profiling logic
    into the training loop.
    """

    def _perform_epoch(self, experiment_results: ExperimentResults, model, device, dataloader, epoch_nr, optimizer):
        profiler = Profiler()
        loss_fn = torch.nn.CrossEntropyLoss()

        is_in_test_mode = optimizer is None
        model.eval() if is_in_test_mode else model.train()

        nr_of_batches = len(dataloader)
        data_iter = iter(dataloader)
        total_loss, acc = 0, 0

        # Warmup GPU to ensure kernels are compiled and ready
        print("Warming up GPU...")
        if nr_of_batches > 0:
            _ = next(data_iter)
            # Reset iterator for actual profiling
            data_iter = iter(dataloader)

        print(f"Profiling Epoch {epoch_nr}...")

        for batch_nr in tqdm(range(nr_of_batches)):
            profiler.start("total_batch")

            # --- 1. DATA LOADING ---
            profiler.start("data_loading")
            try:
                X, y = next(data_iter)
            except StopIteration:
                break
            profiler.end("data_loading")

            # --- PREP ---
            if not is_in_test_mode:
                optimizer.zero_grad()

            y = y.to(device)

            # Note: X transfer happens inside model(X)

            # --- 2. FORWARD PASS ---
            profiler.start("forward")
            predictions = model(X)
            loss = loss_fn(predictions, y)
            profiler.end("forward")

            total_loss += loss.item()

            if not is_in_test_mode:
                # --- 3. BACKWARD PASS ---
                profiler.start("backward")
                loss.backward()
                profiler.end("backward")

                # --- 4. OPTIMIZER STEP ---
                profiler.start("optimizer")
                optimizer.step()
                profiler.end("optimizer")

            y_pred_class = torch.argmax(predictions, dim=1)
            acc += (y_pred_class == y).sum().item() / len(y)

            profiler.end("total_batch")

            # Optional: Stop early to save time, remove if you want full epoch stats
            # if batch_nr >= 100:
            #     print("Stopping early for profiling report...")
            #     break

        profiler.print_summary()

        experiment_results.add_results(epoch_nr, acc, is_in_test_mode)

        return f"no results, profiling only"

    def train_epoch(self, **kwargs):
        return self._perform_epoch(
            experiment_results=kwargs['experiment_results'],
            model=kwargs['model'],
            device=kwargs['device'],
            dataloader=kwargs['dataloader'],
            epoch_nr=kwargs['epoch_nr'],
            optimizer=kwargs['optimizer']
        )

    def test_epoch(self, **kwargs):
        return self._perform_epoch(
            experiment_results=kwargs['experiment_results'],
            model=kwargs['model'],
            device=kwargs['device'],
            dataloader=kwargs['dataloader'],
            epoch_nr=kwargs['epoch_nr'],
            optimizer=None
        )