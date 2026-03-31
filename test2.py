import torch
from torch.optim import Adam
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import torch.nn.functional as F

# -------------------------------
# 1. Setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
).to(device)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler

# -------------------------------
# 2. Add concept tokens
# -------------------------------
token_fg = "<v_star_fg>"
token_bg = "<v_star_bg>"
tokenizer.add_tokens([token_fg, token_bg])
text_encoder.resize_token_embeddings(len(tokenizer))

embedding_dim = text_encoder.config.hidden_size
v_fg = torch.randn(1, embedding_dim, device=device, requires_grad=True)
v_bg = torch.randn(1, embedding_dim, device=device, requires_grad=True)
with torch.no_grad():
    text_encoder.get_input_embeddings().weight[-2] = v_fg
    text_encoder.get_input_embeddings().weight[-1] = v_bg

# -------------------------------
# 3. Load source image
# -------------------------------
source = Image.open("source.png").convert("RGB").resize((512, 512))
source_tensor = torch.from_numpy(np.array(source)/255.0).permute(2,0,1).unsqueeze(0).to(device, dtype=torch.float32)
with torch.no_grad():
    latents = vae.encode(source_tensor).latent_dist.sample() * 0.18215

# -------------------------------
# 4. Load and resize mask
# -------------------------------
mask = Image.open("mask.png").convert("L").resize((512, 512))
mask_fg = torch.from_numpy(np.array(mask)/255.0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
mask_fg = F.interpolate(mask_fg, size=(latents.shape[-2], latents.shape[-1]), mode="nearest")
mask_bg = 1 - mask_fg

# -------------------------------
# 5. Training function
# -------------------------------
def train_token(token_embedding, token_str, mask, num_steps=50, lr=1e-5):
    optimizer = Adam([token_embedding], lr=lr)
    prompt = f"a photo of {token_str}"
    
    for step in range(num_steps):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        embeddings = text_encoder(input_ids).last_hidden_state
        embeddings[:, -1, :] = token_embedding  # assign current token

        # random timestep
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
        noise = torch.randn_like(latents)
        noisy_latent = scheduler.add_noise(latents, noise, t)

        # forward pass
        pred = unet(noisy_latent, t, encoder_hidden_states=embeddings).sample

        # masked loss
        loss = ((pred * mask - noise * mask) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"{token_str} Step {step}, loss = {loss.item():.4f}")

# -------------------------------
# 6. Train foreground token
# -------------------------------
train_token(v_fg, token_fg, mask_fg, num_steps=50, lr=1e-5)

# -------------------------------
# 7. Train background token
# -------------------------------
train_token(v_bg, token_bg, mask_bg, num_steps=50, lr=1e-5)

# -------------------------------
# 8. Generate composed image
# -------------------------------
final_prompt = f"a photo of {token_fg} and {token_bg}"
generated_image = pipe(final_prompt, height=512, width=512, num_inference_steps=25).images[0]
generated_image.save("composed_vstar_fg_bg.png")
generated_image.show()