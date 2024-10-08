import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image

# Disable oneDNN optimizations if needed to avoid issues with low memory
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Ensure 'accelerate' is installed for optimized model loading
try:
    from accelerate import init_empty_weights
except ImportError:
    print("Accelerate library not found. Run 'pip install accelerate' for better performance.")

# Function to generate text using GPT-2
def generate_text(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Function to generate image using Stable Diffusion
def generate_image(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"

    # Check for CUDA availability
    if device == "cuda":
        pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id)

    image = pipe(prompt).images[0]
    return image

if __name__ == "__main__":
    # Define your prompt here
    text_prompt = "A futuristic city skyline at sunset"
    
    print("Generating text...")
    generated_text = generate_text(text_prompt)
    print(f"Generated Text: {generated_text}")
    
    print("Generating image...")
    generated_image = generate_image(text_prompt)
    
    # Save the generated image
    image_path = "generated_image.png"
    generated_image.save(image_path)
    print(f"Image saved to {image_path}")
    
    # Optionally show the image using PIL
    img = Image.open(image_path)
    img.show()

