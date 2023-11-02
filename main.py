from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

from patched_call import patched_call
import clip_image_processor
import clip_text_forward_patch

device='cuda'
dtype = torch.bfloat16

# device='cpu'
# dtype = torch.float32


# loads stable diffusion pipe
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

# loads Human Preference Score v2
clip_model = CLIPModel.from_pretrained("adams-story/HPSv2-hf", torch_dtype=dtype).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") # uses the same exact vanilla clip processor

pipe.safety_checker=None

pipe = pipe.to(device)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

# Use our patched clip_text_forward method
pipe.text_encoder.text_model.forward = lambda *args, **kwargs: clip_text_forward_patch(pipe.text_encoder.text_model, *args, **kwargs)

prompt = "a photo of an apple"

out = patched_call(pipe, prompt, output_type='pt', num_inference_steps = 20,)
image = out.images[0]

# processes image using patched clip processor
clip_image_inputs = clip_image_processor.preprocess(image, clip_processor.image_processor)
clip_text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
clip_text_inputs = {k:v.to(device) for k,v in clip_text_inputs.items()}

clip_outputs = clip_model(pixel_values=clip_image_inputs.unsqueeze(0), **clip_text_inputs)

loss = - clip_outputs.logits_per_image
loss.backward()


# Get the global variable we saved 
hidden_states = clip_text_forward_patch.global_hidden_states


plt.imshow(transforms.ToPILImage()(image), interpolation="bicubic")
plt.savefig("image.jpg")

# Dot product the gradients with the embeddings
# Sum the element wise multiplication across the last dimension
dot_product = (hidden_states * hidden_states.grad).sum(-1) 
# Print the dot product
print("Dot product of gradients with embeddings:", dot_product)
