import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a photo of an exotic landscape with alien flora and fauna, under an unusual sky with multiple moons or distant stars."
image = pipe(prompt).images[0]  
    
image.save("C:/Users/xiaol/Desktop/landscape.png")
