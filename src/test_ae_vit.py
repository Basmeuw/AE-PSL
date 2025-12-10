import os
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from enum import Enum
from PIL import Image
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader, Subset

from available_datasets import CIFAR100
from models.auto_encoder import IdentityAE
from models.meta_transformer.base.data2seq import InputModality
from models.misc.wrapper_model import WrapperModel
from models.vision_transformer.base.ae_vision_transformer import AEVisionTransformer
from utils.mpsl_utils import client_model_requires_any_grad

# --- Setup Environment & SSL ---
# Setting temporary paths for the test execution
os.environ['TORCH_DATA_DIR'] = './data/torch'
os.environ['PRE_PROCESSORS_CACHE_DIR'] = './data/pre_processors_cache'
os.makedirs(os.environ['TORCH_DATA_DIR'], exist_ok=True)
os.makedirs(os.environ['PRE_PROCESSORS_CACHE_DIR'], exist_ok=True)

# Fix SSL context for downloads
ssl._create_default_https_context = ssl._create_unverified_context






# --- Training Loop Function ---
def train_one_epoch(model, loader, optimizer, device, name="Model"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    steps = 0

    print(f"\nTraining {name}...")
    for batch_idx, (inputs, targets) in enumerate(loader):
        # Extract image from the dictionary based on the class structure
        images = inputs[InputModality.IMAGE].to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        # Simple logging
        if batch_idx % 2 == 0:
            print(f"[{name}] Step {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    print(f"[{name}] Finished. Avg Loss: {total_loss / steps:.4f}")


# --- Main Execution ---
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    torch.manual_seed(42)
    # 1. Prepare Dataset
    print("\n--- Initializing Dataset ---")
    # Using a small subset for rapid testing behavior
    full_dataset = CIFAR100(train=True)
    subset_indices = range(0, 32)  # Only use 32 images for the test
    dataset = Subset(full_dataset, subset_indices)

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"Dataset loaded. Batch size: 8. Total Batches: {len(loader)}")

    # ==========================================
    # TEST CASE 1: LoRA + Centralized AE
    # ==========================================
    print("\n" + "=" * 40)
    print("TEST 1: LoRA + Centralized")
    print("=" * 40)

    # Init model with LoRA enabled
    ae = IdentityAE()  # Example AE

    from models.vision_transformer.implementations.unimodal.image_classification.models import get_centralized_model as get_centralized_model_vit
    model_lora = get_centralized_model_vit(auto_encoder=ae, split_layer=3, use_lora=True, lora_rank=4, lora_alpha=4, num_classes=100, device=device)


    model_lora.to(device)

    # Optimizer: In LoRA mode, should optimize only trainable params
    trainable_params = [p for p in model_lora.parameters() if p.requires_grad]
    optimizer_lora = optim.AdamW(trainable_params, lr=1e-3)

    print(f"LoRA Trainable Params: {len(trainable_params)} tensors")
    # train_one_epoch(model_lora, loader, optimizer_lora, device, name="LoRA_Model")

    # ==========================================
    # TEST CASE 2: LoRA + Client and Server AE
    # ==========================================
    print("\n" + "=" * 40)
    print("TEST 2: LoRA + C + S")
    print("=" * 40)

    # Init model with LoRA enabled
    ae = IdentityAE()  # Example AE

    from models.vision_transformer.implementations.unimodal.image_classification.models import \
        get_split_model as get_split_model_vit

    client_model, server_model, client_model_requires_grad = get_split_model_vit(auto_encoder=ae, split_layer=3, use_lora=True, lora_rank=4, lora_alpha=4,
                                           num_classes=100, device=device)
    client_model.to(device)
    server_model.to(device)

    client_model.eval()
    server_model.eval()

    batch = next(iter(loader))
    inputs, targets = batch
    images = inputs[InputModality.IMAGE].to(device)
    targets = targets.to(device)

    with torch.no_grad():
        client_out = client_model(images)
        print(f"client_out shape: {tuple(client_out.shape)}  device: {client_out.device}  dtype: {client_out.dtype}")

        server_out = server_model(client_out)
        print(f"server_out shape: {tuple(server_out.shape)}  device: {server_out.device}  dtype: {server_out.dtype}")

        preds = torch.argmax(server_out, dim=1)
        print(f"preds shape: {tuple(preds.shape)}  targets shape: {tuple(targets.shape)}")

    # wrapper_model = WrapperModel(client_model, server_model)
    # # Forward a batch of data through
    #
    #
    #
    # # Optimizer: In LoRA mode, should optimize only trainable params
    # trainable_params = [p for p in wrapper_model.parameters() if p.requires_grad]
    # optimizer = optim.AdamW(trainable_params, lr=1e-3)
    #
    # print(f"LoRA Trainable Params: {len(trainable_params)} tensors")
    # train_one_epoch(wrapper_model, loader, optimizer, device, name="Full Model")

    # # ==========================================
    # # TEST CASE 1: LoRA + AE
    # # ==========================================
    # print("\n" + "=" * 40)
    # print("TEST 1: LoRA Adaptation Mode")
    # print("=" * 40)
    #
    # # Init model with LoRA enabled
    # ae = AE()  # Example AE
    # model_lora = AEVisionTransformer(auto_encoder=ae, split_layer=3, use_lora=True, lora_rank=4)
    #
    # # IMPORTANT: The pre-trained model has 1000 output classes (ImageNet).
    # # We must replace the head for CIFAR100 (100 classes).
    # # For LoRA mode, we want the head to be trainable.
    # hidden_dim = model_lora.vit.hidden_dim
    # model_lora.vit.heads = nn.Linear(hidden_dim, 100)  # Replaces the head
    #
    # model_lora.to(device)
    #
    # # Optimizer: In LoRA mode, should optimize only trainable params
    # trainable_params = [p for p in model_lora.parameters() if p.requires_grad]
    # optimizer_lora = optim.AdamW(trainable_params, lr=1e-3)
    #
    # print(f"LoRA Trainable Params: {len(trainable_params)} tensors")
    # train_one_epoch(model_lora, loader, optimizer_lora, device, name="LoRA_Model")
    #
    # # ==========================================
    # # TEST CASE 2: Regular Fine-Tuning
    # # ==========================================
    # print("\n" + "=" * 40)
    # print("TEST 2: Full Fine-Tuning Mode")
    # print("=" * 40)
    #
    # # Init model with LoRA disabled
    # model_ft = AEVisionTransformer(auto_encoder=ae, split_layer=3, use_lora=False)
    #
    # # Replace Head
    # model_ft.vit.heads = nn.Linear(hidden_dim, 100)
    #
    # model_ft.to(device)
    #
    # # Optimizer: Optimizes everything
    # optimizer_ft = optim.AdamW(model_ft.parameters(), lr=1e-4)
    #
    # print(f"Full FT Trainable Params: {sum(1 for p in model_ft.parameters() if p.requires_grad)} tensors")
    # train_one_epoch(model_ft, loader, optimizer_ft, device, name="Full_FT_Model")
    #
    # print("\nAll tests completed successfully.")