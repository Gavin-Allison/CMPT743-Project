import torch
from torch.optim import Adam
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from PIL import Image
import numpy as np

# -------------------------------
# 1. Setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler

# -------------------------------
# 2. Add two concept tokens
# -------------------------------
token1_str = "<v_star1>"
token2_str = "<v_star2>"
tokenizer.add_tokens([token1_str, token2_str])
text_encoder.resize_token_embeddings(len(tokenizer))

embedding_dim = text_encoder.config.hidden_size
v_star1 = torch.randn(1, embedding_dim, device=device, requires_grad=True)
v_star2 = torch.randn(1, embedding_dim, device=device, requires_grad=True)
with torch.no_grad():
    text_encoder.get_input_embeddings().weight[-2] = v_star1
    text_encoder.get_input_embeddings().weight[-1] = v_star2

# -------------------------------
# 3. Load source images for each token
# -------------------------------
source1 = Image.open("source1.png").convert("RGB").resize((64, 64))
source2 = Image.open("source2.png").convert("RGB").resize((64, 64))

latents1 = vae.encode(torch.from_numpy(np.array(source1)/255.0).permute(2,0,1).unsqueeze(0).to(device)).latent_dist.sample() * 0.18215
latents2 = vae.encode(torch.from_numpy(np.array(source2)/255.0).permute(2,0,1).unsqueeze(0).to(device)).latent_dist.sample() * 0.18215

# -------------------------------
# 4. Load segmentation masks
# -------------------------------
mask1 = Image.open("mask1.png").convert("L").resize((64, 64))
mask2 = Image.open("mask2.png").convert("L").resize((64, 64))
mask1 = torch.from_numpy(np.array(mask1)/255.0).unsqueeze(0).unsqueeze(0).to(device)
mask2 = torch.from_numpy(np.array(mask2)/255.0).unsqueeze(0).unsqueeze(0).to(device)

# Background mask
mask_bg = 1 - torch.clamp(mask1 + mask2, 0, 1)

# -------------------------------
# 5. Training loop (very small for demo)
# -------------------------------
optimizer = Adam([v_star1, v_star2], lr=1e-4)
prompt = f"a photo of {token1_str} and {token2_str}"
num_steps = 50

for step in range(num_steps):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    embeddings = text_encoder(input_ids).last_hidden_state
    embeddings[:, -2, :] = v_star1
    embeddings[:, -1, :] = v_star2

    t = torch.randint(0, scheduler.num_train_timesteps, (1,), device=device).long()
    noise1 = torch.randn_like(latents1)
    noise2 = torch.randn_like(latents2)

    # Add noise
    noisy_latent1 = scheduler.add_noise(latents1, noise1, t)
    noisy_latent2 = scheduler.add_noise(latents2, noise2, t)

    # Predict noise
    pred1 = unet(noisy_latent1, t, encoder_hidden_states=embeddings).sample
    pred2 = unet(noisy_latent2, t, encoder_hidden_states=embeddings).sample

    # Combine using masks
    combined_pred = pred1 * mask1 + pred2 * mask2 + ((pred1 + pred2)/2) * mask_bg

    # Loss over each region
    loss = ((combined_pred - (noise1 * mask1 + noise2 * mask2)) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, loss = {loss.item():.4f}")

# -------------------------------
# 6. Generate composed image
# -------------------------------
final_prompt = f"a photo of {token1_str} and {token2_str}"
generated_image = pipe(final_prompt, height=64, width=64, num_inference_steps=20).images[0]
generated_image.save("composed_vstar.png")
generated_image.show()