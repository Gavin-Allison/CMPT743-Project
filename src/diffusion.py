import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

class Diffusion_model:
    """
    Takes an image (e.g., stylized output from IP-Adapter) 
    and refines it using Stable Diffusion Img2Img.
    """

    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load SD Img2Img pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32
        ).to(self.device)

    def refine(self, input_image: Image.Image, prompt: str = "", 
               strength: float = 0.2, guidance_scale: float = 1.0, 
               num_inference_steps: int = 40):
        """
        Refine the input image using diffusion.
        """
        with torch.autocast(self.device):
            output = self.pipe(
                prompt=prompt,
                image=input_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
        return output.images[0]
