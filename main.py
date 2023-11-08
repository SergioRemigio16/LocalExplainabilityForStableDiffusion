from typing import List
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

from patched_call import patched_call
import clip_image_processor
import clip_text_forward_patch

device='cuda'
dtype = torch.float32
# device='cpu'
# dtype = torch.float32

# Set seed for deterministic outcome
seed = 16
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
generator = torch.Generator(device=device).manual_seed(seed)

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
pipe.text_encoder.text_model.forward = lambda *args, **kwargs: clip_text_forward_patch.forward(pipe.text_encoder.text_model, *args, **kwargs)

prompt = "A tense, dramatic scene from the television series Peaky Blinders. The air is thick with anticipation as two rival gangs prepare to fight in the dark woods. A cabin stands in the background, providing a looming presence and the potential for escape or refuge. The faces of the gang members are covered in shadows, their features sharp and defined. The image is rendered in black and white, providing a stark and timeless atmosphere. The focus is on the characters' faces, their expressions ranging from determination to fear, all captured in detailed, symmetrical profiles. The camera is positioned to give a medium-long shot, capturing the full breadth of the scene while maintaining a cinematic and epic feel"
prompt = "The artwork portrays a cute goblin girl with a devious expression and a playful air. She has delicate, sharp features and large, alert eyes that seem to radiate curiosity and intelligence. Her ears are large and pointed, a signature feature of goblins. Her hair is a wild, dark mess, adding to her wild and untamed nature."
prompt = "apple photograph black moldy discusting worm-filled rotten "
prompt = "A photograph of a bowl full of berries"


out = patched_call(
    pipe, 
    prompt, 
    output_type='pt', 
    num_inference_steps = 40, 
    guidance_scale = 8.0, 
    eta=1.5,
    generator=generator
)
image = out.images[0]

# processes image using patched clip processor
clip_image_inputs = clip_image_processor.preprocess(image, clip_processor.image_processor)

hps_prompt = prompt
clip_text_inputs = clip_processor(text=[hps_prompt], return_tensors="pt", padding=True, truncation=True)
clip_text_inputs = {k:v.to(device) for k,v in clip_text_inputs.items()}

clip_outputs = clip_model(pixel_values=clip_image_inputs.unsqueeze(0), **clip_text_inputs)

loss = - clip_outputs.logits_per_image
loss.backward()


# Get the global variable we saved 
# the 0th hidden state is the text input embedding
hidden_states = clip_text_forward_patch.global_hidden_states[0]
clip_text_forward_patch.global_hidden_states = []

plt.imshow(transforms.ToPILImage()(image), interpolation="bicubic")
plt.savefig("image.jpg")

input_ids = pipe.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True)['input_ids'][0]

mask_not_padding = input_ids != pipe.tokenizer.pad_token_id

hidden_states_grad = hidden_states.grad[0, mask_not_padding,:]
hidden_states = hidden_states[0, mask_not_padding, :]

# Dot product the gradients with the embeddings
# Sum the element wise multiplication across the last dimension
dot_product = (hidden_states * hidden_states_grad).sum(-1)

dot_product = dot_product - dot_product.min()
dot_product = dot_product / dot_product.sum()
print('sum',dot_product.sum())



for attribution, token_id in zip(dot_product, input_ids):
    print(f'token: "{pipe.tokenizer.decode(token_id):<15}"   attribution: {attribution.item() * 100:6.1f}%')
# Print the dot product
print("Dot product of gradients with embeddings:", dot_product)