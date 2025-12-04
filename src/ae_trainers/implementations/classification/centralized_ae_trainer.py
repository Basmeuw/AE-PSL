from tqdm import tqdm

from src.ae_trainers.implementations.ae_experiment_results import ExperimentResultsAE
from src.trainers.implementations.experiment_trainer import ExperimentTrainer


class CentralizedAETrainer(ExperimentTrainer):
    def _perform_epoch(self, experiment_results: ExperimentResultsAE, base_model, auto_encoder, split_layer, device, dataloader, epoch_nr, optimizer, loss_fn):
        is_in_test_mode = optimizer is None
        auto_encoder.eval() if is_in_test_mode else auto_encoder.train()

        nr_of_batches, data_iter = len(dataloader), iter(dataloader)
        total_loss = 0

        for batch_nr in tqdm(range(nr_of_batches)):
            if not is_in_test_mode:
                optimizer.zero_grad()

            # targets can be ignored, as we use a reconstruction loss
            X, _ = next(data_iter)

            # Note: this currently uses the full model in memory, but forward through the first part only
            # For edge training, we shouldn't pass the full model to all edge devices
            activations = base_model.retrieve_split_layer_activations(X, split_layer)

            reconstructed = auto_encoder(activations)
            loss = loss_fn(reconstructed, activations)

            total_loss += loss.item()

            if not is_in_test_mode:
                loss.backward()
                optimizer.step()


        total_loss /= nr_of_batches


        experiment_results.add_results(epoch_nr, total_loss, is_in_test_mode)

        return f'test loss {total_loss}' if is_in_test_mode else f'Finished epoch {epoch_nr} with train loss {total_loss}'

    def train_epoch(self, **kwargs):
        return self._perform_epoch(
            experiment_results=kwargs['experiment_results'],
            base_model=kwargs['base_model'],
            auto_encoder=kwargs['auto_encoder'],
            split_layer=kwargs['split_layer'],
            device=kwargs['device'],
            dataloader=kwargs['dataloader'],
            epoch_nr=kwargs['epoch_nr'],
            optimizer=kwargs['optimizer'],
            loss_fn=kwargs['loss_fn']
        )

    def test_epoch(self, **kwargs):
        return self._perform_epoch(
            experiment_results=kwargs['experiment_results'],
            base_model=kwargs['base_model'],
            auto_encoder=kwargs['auto_encoder'],
            split_layer=kwargs['split_layer'],
            device=kwargs['device'],
            dataloader=kwargs['dataloader'],
            epoch_nr=kwargs['epoch_nr'],
            optimizer=None,
            loss_fn=kwargs['loss_fn']
        )