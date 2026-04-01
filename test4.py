import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 1. Load base SD model
# -------------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
).to(device)

vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
prompt = ""
text_input = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    encoder_hidden_states = text_encoder(text_input).last_hidden_state

# freeze VAE, allow UNet to train
vae.requires_grad_(False)
unet.requires_grad_(True)

# -------------------------------
# 2. Modify UNet to accept extra channels (FG/BG + mask)
# -------------------------------
old_conv = unet.conv_in
extra_channels = 4 + 4 + 1  # fg latent + bg latent + mask
new_conv = nn.Conv2d(
    in_channels=old_conv.in_channels + extra_channels,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding
)
with torch.no_grad():
    new_conv.weight[:, :old_conv.in_channels] = old_conv.weight
    new_conv.weight[:, old_conv.in_channels:] = 0
    new_conv.bias = old_conv.bias
unet.conv_in = new_conv.to(device)

# -------------------------------
# 3. Load images and mask
# -------------------------------
def load_rgb(path):
    img = Image.open(path).convert("RGB").resize((512,512))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return t.to(device, dtype=torch.float32)

def load_mask(path):
    m = Image.open(path).convert("L").resize((512,512))
    arr = np.array(m).astype(np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    m = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return m.to(device, dtype=torch.float32)

fg_styled = load_rgb("style1.png")  # pre-styled foreground
bg_styled = load_rgb("style2.png")  # pre-styled background
#mask = load_mask("mask.png")
width, height = 512, 512
mask_array = np.zeros((height, width), dtype=np.float32)
mask_array[:, width // 2:] = 1.0
mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

# -------------------------------
# 4. Encode images to latents
# -------------------------------
with torch.no_grad():
    fg_latent = vae.encode(fg_styled).latent_dist.mean * 0.18215
    bg_latent = vae.encode(bg_styled).latent_dist.mean * 0.18215

mask_latent = F.interpolate(mask, size=fg_latent.shape[-2:], mode="nearest")

# Target latent = hard composite
target_latent = fg_latent * mask_latent + bg_latent * (1 - mask_latent)

# -------------------------------
# 5. Training
# -------------------------------
optimizer = Adam(unet.parameters(), lr=1e-5)
num_steps = 1000  # small number for testing

scheduler.set_timesteps(1000)

for step in range(num_steps):
    t = torch.randint(0, scheduler.num_train_timesteps, (1,), device=device).long()
    noise = torch.randn_like(target_latent)
    noisy = scheduler.add_noise(target_latent, noise, t)

    # concat noisy latent + fg/bg + mask
    model_input = torch.cat([noisy, fg_latent, bg_latent, mask_latent], dim=1)

    noise_pred = unet(model_input, t, encoder_hidden_states=encoder_hidden_states).sample

    loss = ((noise_pred - noise) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

# -------------------------------
# 6. Inference
# -------------------------------
latents = torch.randn_like(target_latent)
scheduler.set_timesteps(200)

for t in scheduler.timesteps:
    model_input = torch.cat([latents, fg_latent, bg_latent, mask_latent], dim=1)
    with torch.no_grad():
        noise_pred = unet(model_input, t, encoder_hidden_states=encoder_hidden_states).sample
    latents = scheduler.step(noise_pred, t, latents).prev_sample
# -------------------------------
with torch.no_grad():
    img = vae.decode(latents / 0.18215).sample

img = (img.clamp(-1,1)+1)/2
img = img.cpu().permute(0,2,3,1).numpy()[0]
img = (img*255).astype(np.uint8)

Image.fromarray(img).save("combined_styled.png")
print("Saved output to combined_styled.png")