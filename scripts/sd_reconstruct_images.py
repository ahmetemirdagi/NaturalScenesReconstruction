import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDIMScheduler
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import os



sub = 1  # We're only working with subject 1 for now

print("Loading models...")
# Load the VAE and pipeline
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16
).to("cuda")

pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    vae=vae,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
scheduler.set_timesteps(999)

#pipeline.scheduler = scheduler


# Create output directory if it doesn't exist
output_dir = f'results/stable_diffusion/subj{sub:02d}'
os.makedirs(output_dir, exist_ok=True)

print("Loading predicted features...")
# Load predicted features
pred_features = np.load(f'data/predicted_features/subj{sub:02d}/nsd_cliptext_predtest_nsdgeneral.npy')
init_latents = np.load('data/extracted_features/subj01/nsd_sd_test.npy')

print(pred_features.shape)
print(init_latents.shape)
# Get original shape of latents
original_shape = pred_features.shape
#if len(original_shape) == 2:
    # If features are flattened, reshape them to the VAE latent shape
    # SD VAE latents are typically [B, 4, H/8, W/8] where H,W are image dimensions
    #height, width = 512, 512  # Standard SD image size
    #pred_features = pred_features.reshape(-1, 4, height//8, width//8)

print(f"Reconstructing {len(pred_features)} images...")
# Generate images from latents
for idx in tqdm(range(len(pred_features)), desc="Generating images"):
    # Convert features to torch tensor
    cliptexts = torch.from_numpy(pred_features[idx]).unsqueeze(0).to("cuda", dtype=torch.float16)
    print(cliptexts.shape)
    # Scale the latents (SD expects them scaled)
    # latents = latents * pipeline.vae.config.scaling_factor
    # Latents: assuming `latents` is shape (1, 4, 64, 64)
    latents = torch.from_numpy(init_latents[idx]).unsqueeze(0).to("cuda", dtype=torch.float16) 
    timestep = scheduler.timesteps[49]
    noise = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)
    noisy_latents = scheduler.add_noise(latents, noise, timestep) * vae.config.scaling_factor
    #print(noisy_latents.shape)  
    # Generate image
    with torch.no_grad():
        image = pipeline(
            #latents = noisy_latents,
            #prompt_embeds=cliptexts,
            num_inference_steps=5,
            prompt = 'a man surfing with crazy waves, view from the top',
            guidance_scale = 1.5
                            ).images[0]
    
    # Save the image
    image.save(f'{output_dir}/{idx}.png')

print("\nReconstruction completed! Images saved in:", output_dir)