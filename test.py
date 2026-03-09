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
    torch_dtype=torch.float32
).to(device)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler

# -------------------------------
# 2. Add new concept token
# -------------------------------
concept_token_str = "<v_star>"
tokenizer.add_tokens([concept_token_str])
text_encoder.resize_token_embeddings(len(tokenizer))

embedding_dim = text_encoder.config.hidden_size
v_star = torch.randn(1, embedding_dim, device=device, requires_grad=True)
with torch.no_grad():
    text_encoder.get_input_embeddings().weight[-1] = v_star

# -------------------------------
# 3. Load source image and encode
# -------------------------------
source_img = Image.open("source.png").convert("RGB").resize((512, 512))
source_tensor = torch.from_numpy(np.array(source_img)/255.0).permute(2,0,1).unsqueeze(0).to(device, dtype=torch.float32)

with torch.no_grad():
    latents = vae.encode(source_tensor).latent_dist.sample() * 0.18215

# -------------------------------
# 4. Mini DreamBooth-style training with proper diffusion schedule
# -------------------------------
params_to_optimize = [v_star]
for name, param in unet.named_parameters():
    if "attn" in name:  # fine-tune cross-attention layers
        param.requires_grad = True
        params_to_optimize.append(param)
    else:
        param.requires_grad = False

optimizer = Adam(params_to_optimize, lr=1e-4)
prompt = f"a photo of {concept_token_str}"
num_steps = 500

for step in range(num_steps):
    # 1. Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    embeddings = text_encoder(input_ids).last_hidden_state
    embeddings[:, -1, :] = v_star  # assign learned token

    # 2. Pick random timestep
    t = torch.randint(0, scheduler.num_train_timesteps, (1,), device=device).long()

    # 3. Add noise according to scheduler
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t)

    # 4. Predict noise
    noise_pred = unet(noisy_latents, t, encoder_hidden_states=embeddings).sample

    # 5. Compute L2 loss
    loss = ((noise_pred - noise) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, loss = {loss.item():.4f}")
        torch.save({
            "token_str": "<v_star>",
            "embedding": v_star.detach().cpu()
        }, "v_star.pt")

# -------------------------------
# 5. Generate image using learned token
# -------------------------------
final_prompt = f"a photo of {concept_token_str}"
generated_image = pipe(final_prompt, height=512, width=512, num_inference_steps=50).images[0]
generated_image.save("v_star_full.png")
generated_image.show()