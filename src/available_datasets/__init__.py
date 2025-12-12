import typing

import numpy as np
import numpy.typing as npt
import torch


# Note that this folder is called 'available_datasets' to avoid conflicts with the huggingface 'datasets' package.
from available_datasets.unimodal.image_classification import *
from .multimodal.action_recognition.kinetics_sounds import KineticsSounds
from .multimodal.action_recognition.ucf_101 import UCF101
from .multimodal.emotion_recognition.meld import MELD
from .multimodal.image_text_retrieval.coco_retrieval import CocoRetrieval
from .multimodal.image_text_retrieval.flickr30k import Flickr30K
from .multimodal.sentiment_analysis.t4sa import T4SA
from .multimodal.vqa.coco_qa import COCO_QA
from .unimodal.image_classification.cifar100 import CIFAR100
from .unimodal.image_classification.food101 import Food101

dataloaders = {
	# unimodal
	'cifar100': (CIFAR100, 'cifar100'),
	'food101': (Food101, 'food101'),

    # multimodal
	'coco-qa': (COCO_QA, 'coco-qa'),
	't4sa': (T4SA, 't4sa'),
	'coco-retrieval': (CocoRetrieval, 'coco-retrieval'),
	'flickr30k': (Flickr30K, 'flickr30k'),
	'ucf101': (UCF101, 'ucf101'),
	'kinetics-sounds': (KineticsSounds, 'kinetics-sounds'),
	'meld': (MELD, 'meld')
}


def get_train_val_split(
		dataset,
		val_split,
		seed
):
	"""
    Splits a PyTorch dataset into train and validation subsets using
    torch functions for reproducibility via a torch.Generator.
    """
	if val_split == 0.0:
		# If no validation split is requested, return the full dataset as train
		# and None for the validation set.
		return Subset(dataset, torch.arange(len(dataset))), None

	if not (0.0 < val_split < 1.0):
		raise ValueError("val_split must be between 0.0 and 1.0")

	dataset_len = len(dataset)
	# Calculate lengths
	val_len = int(dataset_len * val_split)
	train_len = dataset_len - val_len

	# --- Use torch for reproducible shuffling ---
	# 1. Create a torch.Generator instance with the specified seed.
	generator = torch.Generator().manual_seed(seed)

	# 2. Generate a random permutation of indices from 0 to dataset_len - 1.
	# The shuffling is controlled by the 'generator'.
	shuffled_indices = torch.randperm(dataset_len, generator=generator)

	# 3. Split the shuffled indices
	train_indices = shuffled_indices[:train_len].tolist()
	val_indices = shuffled_indices[train_len:].tolist()

	# 4. Create Subsets
	train_subset = Subset(dataset, train_indices)
	val_subset = Subset(dataset, val_indices)

	return train_subset, val_subset

class Subset(torch.utils.data.Subset):

	@property
	def num_classes(self):
		return self.dataset.num_classes

	@property
	def targets(self):
		"""
        Expose targets so DirichletDataPartitioner can access them from the Subset.
        Assumes the underlying dataset has a 'targets' list/array.
        """
		# Access the underlying targets using the subset indices
		return [self.dataset.targets[i] for i in self.indices]

	@property
	def num_rows(self):
		"""Expose num_rows (length) for compatibility with the partitioner checks."""
		return len(self)


class DataLoader(torch.utils.data.DataLoader):

	@property
	def num_classes(self):
		return self.dataset.num_classes




class DirichletDataPartitioner:
	"""
	As per https://github.com/adap/flower
	"""

	def __init__(
			self,
			dataset: torch.utils.data.Dataset,
			num_partitions: int,
			alpha: typing.Union[int, float, typing.List[float], np.ndarray],
			min_partition_size: int = 10,
			self_balancing: bool = False,
			shuffle: bool = True,
			seed: typing.Optional[int] = 42,
			custom_partitioner_function=None
	) -> None:

		self.dataset = dataset
		self._num_partitions = num_partitions
		self._check_num_partitions_greater_than_zero()
		self._alpha = self._initialize_alpha(alpha)
		self._min_partition_size: int = min_partition_size
		self._self_balancing = self_balancing
		self._shuffle = shuffle
		self._seed = seed
		self._rng = np.random.default_rng(seed=self._seed)
		
		# Utility attributes
		self._avg_num_of_samples_per_partition: typing.Optional[float] = None
		self._unique_classes: typing.Optional[typing.Union[typing.List[int], typing.List[str]]] = None
		self._partition_id_to_indices: typing.Dict[int, typing.List[int]] = {}
		self._partition_id_to_indices_determined = False

		# Some datasets require separate code for partitioning, for which we need to override default behavior.
		self.custom_partitioner_function = custom_partitioner_function

	@property
	def num_partitions(self) -> int:
		self._check_num_partitions_correctness_if_needed()
		self._determine_partition_id_to_indices_if_needed()

		return self._num_partitions

	def _check_num_partitions_greater_than_zero(self) -> None:
		if not self._num_partitions > 0:
			raise ValueError("The number of partitions needs to be greater than zero.")

	def _check_num_partitions_correctness_if_needed(self) -> None:
		if not self._partition_id_to_indices_determined:
			if self._num_partitions > self.dataset.num_rows:
				raise ValueError("The number of partitions needs to be smaller than the number of samples in the dataset.")
				
	def _initialize_alpha(self, alpha: typing.Union[int, float, typing.List[float], npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
		if isinstance(alpha, int):
			alpha = np.array([float(alpha)], dtype=float).repeat(self._num_partitions)
		elif isinstance(alpha, float):
			alpha = np.array([alpha], dtype=float).repeat(self._num_partitions)
		elif isinstance(alpha, typing.List):
			if len(alpha) != self._num_partitions:
				raise ValueError("If passing alpha as a List, it needs to be of length of equal to num_partitions.")
			alpha = np.asarray(alpha)
		elif isinstance(alpha, np.ndarray):
			if alpha.ndim == 1 and alpha.shape[0] != self._num_partitions:
				raise ValueError("If passing alpha as an NDArray, its length needs to be of length equal to num_partitions.")
			elif alpha.ndim == 2:
				alpha = alpha.flatten()
				if alpha.shape[0] != self._num_partitions:
					raise ValueError("If passing alpha as an NDArray, its size needs to be of length equal to num_partitions.")
		else:
			raise ValueError("The given alpha format is not supported.")
		if not (alpha > 0).all():
			raise ValueError(f"Alpha values should be strictly greater than zero. Instead it'd be converted to {alpha}")
		return alpha

	def _determine_partition_id_to_indices_if_needed(self) -> None:
		if self._partition_id_to_indices_determined:
			return

		if self.custom_partitioner_function is not None:
			self._partition_id_to_indices = self.custom_partitioner_function(self._num_partitions, len(self.dataset), shuffle_indices=True)
		else:
			# Generate information needed for Dirichlet partitioning
			targets = np.asarray(self.dataset.targets) #np.array([self.dataset[i][1] for i in range(len(self.dataset))])
			self._unique_classes = np.unique(targets).tolist()
			assert self._unique_classes is not None
			self._avg_num_of_samples_per_partition = len(self.dataset) / self._num_partitions

			# Repeat the sampling procedure based on the Dirichlet distribution until the min_partition_size is reached.
			sampling_try = 0
			while True:
				# Prepare data structure to store indices assigned to partition ids
				partition_id_to_indices: typing.Dict[int, typing.List[int]] = {nid: [] for nid in range(self._num_partitions)}
				# Iterate over all unique labels
				for k in self._unique_classes:
					# Access all the indices associated with class k
					indices_representing_class_k = np.where(targets == k)[0]
					# Determine division (the fractions) of the data representing class k among the partitions
					class_k_division_proportions = self._rng.dirichlet(self._alpha)
					nid_to_proportion_of_k_samples = {nid: class_k_division_proportions[nid] for nid in range(self._num_partitions)}
					# Balancing
					if self._self_balancing:
						assert self._avg_num_of_samples_per_partition is not None
						for nid in nid_to_proportion_of_k_samples.copy():
							if len(partition_id_to_indices[nid]) > self._avg_num_of_samples_per_partition:
								nid_to_proportion_of_k_samples[nid] = 0
						sum_proportions = sum(nid_to_proportion_of_k_samples.values())
						for nid in nid_to_proportion_of_k_samples:
							nid_to_proportion_of_k_samples[nid] /= sum_proportions
					# Determine the split indices
					cumsum_division_fractions = np.cumsum(list(nid_to_proportion_of_k_samples.values()))
					cumsum_division_numbers = cumsum_division_fractions * len(indices_representing_class_k)
					indices_on_which_split = cumsum_division_numbers.astype(int)[:-1]
					split_indices = np.split(indices_representing_class_k, indices_on_which_split)
					# Append new indices (coming from class k) to the existing indices
					for nid in range(self._num_partitions):
						partition_id_to_indices[nid].extend(split_indices[nid].tolist())
				# Check if the indices assignment meets the min_partition_size
				min_sample_size_on_client = min(len(indices) for indices in partition_id_to_indices.values())
				if min_sample_size_on_client >= self._min_partition_size:
					break
				sampling_try += 1
				if sampling_try == 10:
					raise ValueError("The max number of attempts (10) was reached. Please update the values of alpha and try again.")

			# Shuffle the indices if shuffle is True
			if self._shuffle:
				for indices in partition_id_to_indices.values():
					self._rng.shuffle(indices)
			self._partition_id_to_indices = partition_id_to_indices

		self._partition_id_to_indices_determined = True

	def load_partition(self, partition_id: int) -> torch.utils.data.Dataset:
		self._determine_partition_id_to_indices_if_needed()
		indices = self._partition_id_to_indices[partition_id]

		return Subset(self.dataset, indices)

	@property
	def num_partitions(self) -> int:
		self._determine_partition_id_to_indices_if_needed()

		return self._num_partitions


class DistributedDataset(torch.utils.data.Dataset):
	"""
    As per FederatedDataset of https://github.com/adap/flower
    """

	def __init__(
			self,
			dataloader,
			transform=None,
			alpha=0.5,
			num_partitions=10,
			min_partition_size=10,
			self_balancing=True,
			shuffle=False,
			seed=42,
			val_split=0.0,  # <--- NEW PARAMETER
			**kwargs,
	):
		self._dataloader = dataloader
		self._dataloader_args = {'transform': transform, 'seed': seed, 'global_args': kwargs['global_args']}
		self._dataloader_args.update(kwargs)

		# 1. Load the FULL training dataset
		full_train_ds = self._dataloader(
			train=True,
			**self._dataloader_args
		)

		# 2. Split into Train and Validation (before partitioning)
		self.train_ds, self.val_ds = get_train_val_split(full_train_ds, val_split, seed)

		custom_partitioner_function = self.train_ds.custom_partitioner_function if hasattr(self.train_ds,
																						   'custom_partitioner_function') else None

		# 3. Create data partitioner using ONLY the training subset
		self.partitioner = DirichletDataPartitioner(
			dataset=self.train_ds,
			num_partitions=num_partitions,
			alpha=alpha,
			min_partition_size=min_partition_size,
			self_balancing=self_balancing,
			shuffle=shuffle,
			seed=seed,
			custom_partitioner_function=custom_partitioner_function
		)

		self._seed = seed

	# ... [keep existing properties: num_classes, classes, __getitem__, __len__, load_partition] ...

	@property
	def num_classes(self):
		return self.train_ds.num_classes

	@property
	def classes(self):
		# Handle case where Subset might not have _classes directly if needed,
		# but usually accessing the underlying dataset is safer if _classes is not on Subset
		return list(self.train_ds.dataset._classes.keys()) if hasattr(self.train_ds, 'dataset') else list(
			self.train_ds._classes.keys())

	def __getitem__(self, idx):
		return self.train_ds[idx]

	def __len__(self):
		return len(self.train_ds)

	def load_partition(self, partition_id: int) -> torch.utils.data.Dataset:
		partition = self.partitioner.load_partition(partition_id)
		return partition

	# ... [keep load_test_set] ...
	def load_test_set(self) -> torch.utils.data.Dataset:
		test_ds = self._dataloader(
			train=False,
			**self._dataloader_args
		)
		return test_ds

	# 4. New method to access the global validation set
	def load_validation_set(self) -> torch.utils.data.Dataset:
		if self.val_ds is None:
			return None
			# raise ValueError("No validation split was created. Initialize with val_split > 0.")
		return self.val_ds

	def get_collate_fn(self):
		return self._dataloader.custom_collate_fn


def available_datasets():
	return list(dataloaders.keys())


def load_data(name='cifar100', num_partitions=10, min_num_samples=10, split='iid', seed=42, transform=None, global_args=None, val_split=0.0):
    assert name in dataloaders.keys(), 'Dataset `{}` is not available. Available datasets are `{}`'.format(name, available_datasets())

    if split == 'iid':
       alpha = 100000.0
    elif split == 'noniid':
       alpha = 0.5
    elif isinstance(split, int):
       alpha = split
    else:
       ValueError('`split` can be either `iid`, `noniid`, `int` or `float`. Passed {} of type {}'.format(split, type(split)))

    ds, folder_name = dataloaders[name]

    return DistributedDataset(
       dataloader=ds,
       transform=transform,
       alpha=alpha,
       num_partitions=num_partitions,
       min_partition_size=min_num_samples,
       self_balancing=True,
       shuffle=False,
       seed=seed,
       global_args=global_args,
       val_split=val_split # <--- Pass it here
    )
