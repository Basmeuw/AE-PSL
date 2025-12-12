# Similar to MPSL Trainer, but for unimodal models

import torch
from tqdm import tqdm

from models.meta_transformer.base.data2seq import InputModality
from trainers.implementations.experiment_results import ExperimentResults
from trainers.implementations.experiment_trainer import ExperimentTrainer
from utils.mpsl_utils import get_communication_size, compute_aggregated_loss, bytes_to_megabytes, get_client_name

loss_fn = torch.nn.CrossEntropyLoss()


class SASLTrainer(ExperimentTrainer):

    def __init__(self):
        super()


    def _single_epoch(self, device, server_model, server_optimizer, client_data_iterators: dict, client_models: dict, client_model_requires_any_grad, client_optimizers: dict, max_nr_of_batches_in_epoch, global_args):
        client_ids = range(global_args['nr_of_clients'])
        total_server_loss, acc = 0, 0
        client_specific_acc_tuples, nr_of_elements_per_client = {_client_id: (0, 0) for _client_id in client_ids}, {_client_id: 0 for _client_id in client_ids}

        # = Communication tracking =
        client_outgoing_communication_sizes, client_incoming_communication_sizes = {_client_id: 0 for _client_id in client_ids}, {_client_id: 0 for _client_id in client_ids}
        server_outgoing_communication_size, server_incoming_communication_size = 0, 0

        for batch_nr in tqdm(range(max_nr_of_batches_in_epoch)):
            server_optimizer.zero_grad()

            # We combine the activations of all client mini-batches into a single large batch to speed up training, as it only requires a single forward pass on the server.
            intermediate_activations_entire_batch_combined = None
            intermediate_activations_per_client, client_loss_fn_tuples = dict(), []

            # The indices that correspond to each client's mini-batch, given that the full batch is the concatenation of all mini-batches together
            mini_batch_indices, y_per_client = dict(), dict()

            for client_id in range(global_args['nr_of_clients']):
                try:
                    X, y = next(client_data_iterators[client_id])

                    # Intentionally only sending labels to the device here. Depending on whether input is multimodal and the modalities themselves, input has to be handled differently
                    # prior to sending (chunks of) it to the device.
                    y = y.to(device)
                    y_per_client[client_id] = y

                    nr_of_elements_per_client[client_id] += len(y)
                except StopIteration as exception:
                    # When this error is thrown, the iterator has no remaining elements, which might occur when some clients have more batches than others.
                    print(exception)
                    continue

                # FP on client-side model
                client_optimizers[client_id].zero_grad()
                intermediate_activations = client_models[client_id](X)
                intermediate_activations_per_client[client_id] = intermediate_activations

                # = Communication tracking =
                comms_size = get_communication_size(intermediate_activations)
                client_outgoing_communication_sizes[client_id] += comms_size
                server_incoming_communication_size += comms_size

                # Combine all intermediate activations to allow for a single FP
                if intermediate_activations_entire_batch_combined is None:
                    intermediate_activations_entire_batch_combined = intermediate_activations

                    mini_batch_indices[client_id] = (0, len(intermediate_activations))
                else:
                    client_begin_index = len(intermediate_activations_entire_batch_combined)
                    mini_batch_indices[client_id] = (client_begin_index, client_begin_index + len(y))

                    intermediate_activations_entire_batch_combined = torch.cat((intermediate_activations_entire_batch_combined, intermediate_activations))

            # Single full batch FP on server-side model
            final_activations = intermediate_activations_entire_batch_combined.detach().clone().requires_grad_(True)
            predictions = server_model(final_activations)

            for client_id in range(global_args['nr_of_clients']):
                if client_id not in mini_batch_indices:
                    continue

                # Loss computation on client-side
                client_mini_batch_indices = mini_batch_indices[client_id]
                preds_for_client = predictions[client_mini_batch_indices[0]:client_mini_batch_indices[1]]
                y_for_client = y_per_client[client_id]
                mini_batch_length = len(y_for_client)

                client_loss_fn = loss_fn(preds_for_client, y_for_client)
                client_loss_fn_tuples.append((client_loss_fn, mini_batch_length))

                # = Communication tracking =
                comms_size = get_communication_size(preds_for_client)
                client_incoming_communication_sizes[client_id] += comms_size
                server_outgoing_communication_size += comms_size

                comms_size = get_communication_size(client_loss_fn)
                client_outgoing_communication_sizes[client_id] += comms_size
                server_incoming_communication_size += comms_size

                # Metrics
                y_pred_class = torch.argmax(preds_for_client, dim=1)
                current_batch_acc = (y_pred_class == y_for_client).sum().item() / mini_batch_length
                acc += current_batch_acc

                client_specific_acc, client_specific_total_batch_size = client_specific_acc_tuples[client_id]
                client_specific_acc_tuples[client_id] = (client_specific_acc + current_batch_acc, client_specific_total_batch_size + 1)

            # BP on server-side
            agg_loss = compute_aggregated_loss(client_loss_fn_tuples, len(intermediate_activations_entire_batch_combined))

            total_server_loss += agg_loss.item()

            agg_loss.backward()

            server_optimizer.step()

            # We should only compute gradients and perform client-side BP if client-side models actually require gradients.
            if client_model_requires_any_grad:
                # Server-side 'sending' gradients & client-side performing BP
                for client_id in range(global_args['nr_of_clients']):
                    if client_id in mini_batch_indices:
                        (client_begin_index, client_end_index) = mini_batch_indices[client_id]
                        cut_layer_grads = final_activations.grad[client_begin_index:client_end_index].clone()

                        intermediate_activations_per_client[client_id].backward(cut_layer_grads)
                        client_optimizers[client_id].step()

                        # = Communication tracking =
                        comms_size = get_communication_size(cut_layer_grads)
                        client_incoming_communication_sizes[client_id] += comms_size
                        server_outgoing_communication_size += comms_size

        return total_server_loss, \
            acc, \
            server_incoming_communication_size, \
            server_outgoing_communication_size, \
            client_incoming_communication_sizes, \
            client_outgoing_communication_sizes, \
            client_specific_acc_tuples, \
            nr_of_elements_per_client


    def _train_epoch(self, device, server_model, server_optimizer, client_data_iterators, client_models, client_model_requires_any_grad, client_optimizers, max_nr_of_batches_in_epoch, client_schedulers, experiment_results: ExperimentResults, epoch_nr, global_args):
        """
        We are using activations combining; We combine all (modality-specific) client-side activations into a single large batch, requiring only a single FP on the server-model. This speeds up training.
        """
        [model.train() for model in client_models.values()]
        server_model.train()

        total_server_loss, \
        acc, \
        server_incoming_communication_size, \
        server_outgoing_communication_size, \
        client_incoming_communication_sizes, \
        client_outgoing_communication_sizes, \
        client_specific_acc_tuples, \
        nr_of_elements_per_client_dict = self._single_epoch(
            device,
            server_model,
            server_optimizer,
            client_data_iterators,
            client_models,
            client_model_requires_any_grad,
            client_optimizers,
            max_nr_of_batches_in_epoch,
            global_args)

        total_server_loss /= max_nr_of_batches_in_epoch
        acc = ((acc / max_nr_of_batches_in_epoch) / global_args['nr_of_clients'])

        experiment_results.add_results(epoch_nr, acc, False)

        print(f'Finished training epoch with server communication overhead: incoming {bytes_to_megabytes(server_incoming_communication_size)} MB & outgoing {bytes_to_megabytes(server_outgoing_communication_size)} MB')

        total_client_outgoing_communication_size, total_client_incoming_communication_size = 0, 0

        for client_id in range(global_args['nr_of_clients']):
            client_specific_total_acc, client_specific_total_batch_size = client_specific_acc_tuples[client_id]

            # = Communication tracking =
            client_outgoing_comms, client_incoming_comms = client_outgoing_communication_sizes[client_id], client_incoming_communication_sizes[client_id]
            total_client_outgoing_communication_size += client_outgoing_comms
            total_client_incoming_communication_size += client_incoming_comms

            print(f'Client-specific train accuracy for {get_client_name(client_id)}: {client_specific_total_acc / client_specific_total_batch_size} with communication overhead: incoming {bytes_to_megabytes(client_incoming_comms)} MB & outgoing {bytes_to_megabytes(client_outgoing_comms)} MB')

            if client_model_requires_any_grad:
                client_schedulers[client_id].step()

        # = Communication tracking =
        avg_incoming_comms_overhead_in_mb = bytes_to_megabytes(total_client_incoming_communication_size / global_args['nr_of_clients'])
        avg_outgoing_comms_overhead_in_mb = bytes_to_megabytes(total_client_outgoing_communication_size / global_args['nr_of_clients'])
        print(f'Average client communication overhead: incoming {avg_incoming_comms_overhead_in_mb} MB & outgoing {avg_outgoing_comms_overhead_in_mb} MB')
        experiment_results.set_client_communication_overhead(avg_incoming_comms_overhead_in_mb, avg_outgoing_comms_overhead_in_mb)

        return total_server_loss, acc, nr_of_elements_per_client_dict

    def _test_epoch(self, client_model, server_model, dataloader, device, experiment_results: ExperimentResults, epoch_nr):
        client_model.eval()
        server_model.eval()

        total_server_loss, acc = 0, 0

        nr_of_batches, data_iter = len(dataloader), iter(dataloader)

        with torch.no_grad():
            for batch_nr in tqdm(range(nr_of_batches)):
                try:
                    X, y = next(data_iter)
                except StopIteration:
                    # When this error is thrown, the iterator has no remaining elements, which might occur when some clients have more batches than others.
                    continue

                # Intentionally only sending labels to the device here. Depending on whether input is multimodal and the modalities themselves, input has to be handled differently
                # prior to sending (chunks of) it to the device.
                y = y.to(device)

                # FP on client-side and model followed by FP on server-side model
                predictions = server_model(client_model(X))

                # Loss computation on client-side
                client_loss = loss_fn(predictions, y)

                # Metrics
                y_pred_class = torch.argmax(predictions, dim=1)
                current_batch_acc = (y_pred_class == y).sum().item() / len(y)
                acc += current_batch_acc

                # BP on server-side
                batch_size = len(y)
                agg_loss = compute_aggregated_loss([(client_loss, batch_size)], batch_size)

                total_server_loss += agg_loss.item()

        total_server_loss /= nr_of_batches
        acc = acc / nr_of_batches

        experiment_results.add_results(epoch_nr, acc, True)

        return total_server_loss, acc

    def train_epoch(self, **kwargs):
        return self._train_epoch(
            device=kwargs['device'],
            server_model=kwargs['server_model'],
            server_optimizer=kwargs['server_optimizer'],
            client_data_iterators=kwargs['client_data_iterators'],
            client_models=kwargs['client_models'],
            client_model_requires_any_grad=kwargs['client_model_requires_any_grad'],
            client_optimizers=kwargs['client_optimizers'],
            max_nr_of_batches_in_epoch=kwargs['max_nr_of_batches_in_epoch'],
            client_schedulers=kwargs['client_schedulers'],
            experiment_results=kwargs['experiment_results'],
            epoch_nr=kwargs['epoch_nr'],
            global_args=kwargs['global_args']
        )

    def test_epoch(self, **kwargs):
        return self._test_epoch(
            client_model=kwargs['client_model'],
            server_model=kwargs['server_model'],
            dataloader=kwargs['dataloader'],
            device=kwargs['device'],
            experiment_results=kwargs['experiment_results'],
            epoch_nr=kwargs['epoch_nr']
        )
