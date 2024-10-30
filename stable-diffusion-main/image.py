from diffusers import DiffusionPipeline
import torch
from PIL import Image
import numpy as np

# 检查GPU是否可用
if torch.cuda.is_available():
    device = "cuda"
else:
    raise RuntimeError("CUDA not available. Please check your GPU setup.")

# Load base diffusion pipeline, use FP16 to save memory
base_pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
).to(device)

 # Define generation prompts and parameters
prompt = "a photo of an astronaut riding a green horse."
num_inference_steps = 20  # Reducing the number of reasoning steps
guidance_scale = 7.0  # Reduced guidance size to reduce memory footprint

final_image = base_pipe(
    prompt=prompt, 
    num_inference_steps=num_inference_steps, 
    guidance_scale=guidance_scale, 
    output_type="pil"  
).images[0]

img_array = np.array(final_image)  

final_image = Image.fromarray(img_array) 

output_path = "C:/Users/xiaol/Desktop/output_image1.png"
final_image.save(output_path)

print(f"Image generated and saved to: {output_path}")

import subprocess
gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print("GPU information:", gpu_info.stdout)
