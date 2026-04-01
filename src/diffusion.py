import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

class LatentCompositeDiffusion:
    """
    Refines and combines pre-styled foreground and background images
    using a modified Stable Diffusion UNet (extra channels for fg/bg/mask).
    """

    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load base SD model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(self.device)

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

        # Freeze VAE, UNet trainable
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(True)

        # Extend UNet for extra channels (fg/bg + mask)
        self._extend_unet_channels(extra_channels=4 + 4 + 1)

    def _extend_unet_channels(self, extra_channels: int):
        old_conv = self.unet.conv_in
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
        self.unet.conv_in = new_conv.to(self.device)

    def train(self, fg_img, bg_img, mask, prompt="", 
              lr=1e-5, steps=1000):
        device = self.device
        
        fg_img = fg_img.convert("RGB")
        arr = np.array(fg_img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        fg_tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device, dtype=torch.float32)
        
        bg_img = bg_img.convert("RGB")
        arr = np.array(bg_img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        bg_tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device, dtype=torch.float32)

        fg_latent = self.vae.encode(fg_tensor).latent_dist.mean * 0.18215
        bg_latent = self.vae.encode(bg_tensor).latent_dist.mean * 0.18215

        mask = mask / mask.max()
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        mask_latent = F.interpolate(mask, size=fg_latent.shape[-2:], mode="bilinear", align_corners=False)
        mask_latent = F.avg_pool2d(mask_latent, kernel_size=3, stride=1, padding=1)

        # Hard composite target
        target_latent = fg_latent * mask_latent + bg_latent * (1 - mask_latent)

        # Text embedding
        text_input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(text_input).last_hidden_state

        optimizer = Adam(self.unet.parameters(), lr=lr)
        self.scheduler.set_timesteps(1000)

        for step in range(steps):
            t = torch.randint(0, self.scheduler.num_train_timesteps, (1,), device=device).long()
            noise = torch.randn_like(target_latent)
            noisy = self.scheduler.add_noise(target_latent, noise, t)

            model_input = torch.cat([noisy, fg_latent, bg_latent, mask_latent], dim=1)
            noise_pred = self.unet(model_input, t, encoder_hidden_states=encoder_hidden_states).sample

            loss = ((noise_pred - noise)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

        self.fg_latent = fg_latent
        self.bg_latent = bg_latent
        self.mask_latent = mask_latent
        self.encoder_hidden_states = encoder_hidden_states

    def generate(self, steps=200):
        latents = torch.randn_like(self.fg_latent)
        self.scheduler.set_timesteps(steps)
        for t in self.scheduler.timesteps:
            model_input = torch.cat([latents, self.fg_latent, self.bg_latent, self.mask_latent], dim=1)
            with torch.no_grad():
                noise_pred = self.unet(model_input, t, encoder_hidden_states=self.encoder_hidden_states).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        with torch.no_grad():
            img = self.vae.decode(latents / 0.18215).sample

        img = (img.clamp(-1,1)+1)/2
        img = img.cpu().permute(0,2,3,1).numpy()[0]
        img = (img*255).astype(np.uint8)
        return Image.fromarray(img)