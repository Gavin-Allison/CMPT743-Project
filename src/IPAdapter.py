import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


class IPAdapter:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # load IP-Adapter weights
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )

        # how strongly style image influences output
        self.pipe.set_ip_adapter_scale(0.7)

    def stylize(self, content, style):

        result = self.pipe(
            prompt="",
            image=content,
            ip_adapter_image=style,
            strength=0.6,
            num_inference_steps=20
        ).images[0]

        return result