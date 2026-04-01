import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 1. Load Stable Diffusion
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

# -------------------------------
# 2. Load images and mask
# -------------------------------
def load_rgb(path):
    img = Image.open(path).convert("RGB").resize((512,512))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return t.to(device, dtype=torch.float32)

def load_mask(path=None):
    # simple default mask: left=BG, right=FG
    width, height = 512, 512
    mask_array = np.zeros((height, width), dtype=np.float32)
    mask_array[:, width // 2:] = 1.0
    m = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)
    return m.to(device, dtype=torch.float32)

fg_styled = load_rgb("style1.png")
bg_styled = load_rgb("style2.png")
mask = load_mask()

# -------------------------------
# 3. Encode to VAE latents
# -------------------------------
with torch.no_grad():
    fg_latent = vae.encode(fg_styled).latent_dist.mean * 0.18215
    bg_latent = vae.encode(bg_styled).latent_dist.mean * 0.18215

mask_latent = F.interpolate(mask, size=fg_latent.shape[-2:], mode="nearest")

# -------------------------------
# 4. Hard latent composite
# -------------------------------
combined_latent = fg_latent * mask_latent + bg_latent * (1 - mask_latent)

# -------------------------------
# 5. SD refinement (denoising)
# -------------------------------
latents = combined_latent.clone()
num_steps = 50
scheduler.set_timesteps(num_steps)

# Empty prompt for neutral refinement (can add text if you want)
prompt = ""
text_input = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    encoder_hidden_states = text_encoder(text_input).last_hidden_state

for t in scheduler.timesteps:
    with torch.no_grad():
        noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# -------------------------------
# 6. Decode final image
# -------------------------------
with torch.no_grad():
    img = vae.decode(latents / 0.18215).sample

img = (img.clamp(-1,1)+1)/2
img = img.cpu().permute(0,2,3,1).numpy()[0]
img = (img*255).astype(np.uint8)
Image.fromarray(img).save("combined_styled_refined.png")
print("Saved output to combined_styled_refined.png")