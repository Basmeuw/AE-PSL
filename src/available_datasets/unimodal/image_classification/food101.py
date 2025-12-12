import os
# Disable SSL verification for the downloading of the datasets
import ssl

import PIL
import torch
import torchvision
from PIL import Image
from transformers import AutoImageProcessor

from models.meta_transformer.base.data2seq import InputModality

ssl._create_default_https_context = ssl._create_unverified_context


class Food101(torchvision.datasets.Food101):

	custom_collate_fn = None

	def __init__(self, train=False, transform=None, **kwargs):

		super().__init__(os.path.join(os.environ['TORCH_DATA_DIR'], 'food101'), split="train" if train else "test", transform=transform, target_transform=None, download=True)
		# self.data_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", cache_dir=os.path.join(os.environ['PRE_PROCESSORS_CACHE_DIR'], 'vit-mae-base'))
		self.transform = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()
		self.data_processor = None

	@property
	def num_classes(self):
		return 101

	@property
	def targets(self):
		# accessing a private variable here
		return self._labels

	def __getitem__(self, idx):
		image_file, label = self._image_files[idx], self._labels[idx]
		image = PIL.Image.open(image_file).convert("RGB")

		if self.transform:
			image = self.transform(image)

		if self.target_transform:
			label = self.target_transform(label)

		return  {InputModality.IMAGE: image}, label

	def __len__(self):
		return len(self.targets)
