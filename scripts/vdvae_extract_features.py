import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKL
from tqdm import tqdm

print("Starting script...")

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Initialize the SD VAE model
print("Loading VAE model...")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = vae.to(device)
print("VAE loaded successfully")

class batch_generator_external_images(Dataset):
    def __init__(self, data_path):
        print(f"Loading data from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)
        print(f"Loaded data shape: {self.im.shape}")

    def __getitem__(self, idx):
        img = Image.fromarray(self.im[idx])
        # Resize to 512x512 as expected by SD VAE
        img = T.functional.resize(img, (512, 512))
        img = T.functional.to_tensor(img).float()
        # Normalize to [-1, 1] range as expected by SD VAE
        img = img * 2.0 - 1.0
        return img

    def __len__(self):
        return len(self.im)

# Create output directory if it doesn't exist
output_dir = 'data/extracted_features/subj01'
if not os.path.exists(output_dir):
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

# Setup data loaders
batch_size = 8  # Increased batch size for 3090
print("\nSetting up data loaders...")

image_path = 'data/processed_data/subj01/nsd_test_stim_sub1.npy'
print(f"Loading test data from: {image_path}")
test_images = batch_generator_external_images(data_path=image_path)
testloader = DataLoader(test_images, batch_size, shuffle=False, num_workers=4)
print(f"Test loader created with {len(testloader)} batches")

print("Extracting features...")

# Extract features for test set
test_features = []
with torch.no_grad():
    for x in tqdm(testloader, desc="Processing test images", total=len(testloader)):
        x = x.to(device, dtype=torch.float16)
        # Get VAE features (latent representation)
        latents = vae.encode(x).latent_dist.sample()
        # Scale latents as done in Stable Diffusion
        latents = latents * vae.config.scaling_factor
        test_features.append(latents.cpu().numpy())

test_features = np.concatenate(test_features)
print(f"\nSaving test features to: {output_dir}/nsd_sd_test.npy")
np.save(f'{output_dir}/nsd_sd_test.npy', test_features)
print(f"Test features shape: {test_features.shape}")

# Now load training data
image_path = 'data/processed_data/subj01/nsd_train_stim_sub1.npy'
print(f"\nLoading training data from: {image_path}")
train_images = batch_generator_external_images(data_path=image_path)
trainloader = DataLoader(train_images, batch_size, shuffle=False, num_workers=4)
print(f"Train loader created with {len(trainloader)} batches")

# Extract features for training set
train_features = []
with torch.no_grad():
    for x in tqdm(trainloader, desc="Processing train images", total=len(trainloader)):
        x = x.to(device, dtype=torch.float16)
        # Get VAE features (latent representation)
        latents = vae.encode(x).latent_dist.sample()
        # Scale latents as done in Stable Diffusion
        latents = latents * vae.config.scaling_factor
        train_features.append(latents.cpu().numpy())

train_features = np.concatenate(train_features)
print(f"\nSaving train features to: {output_dir}/nsd_sd_train.npy")
np.save(f'{output_dir}/nsd_sd_train.npy', train_features)
print(f"Train features shape: {train_features.shape}")

print("Feature extraction completed!")