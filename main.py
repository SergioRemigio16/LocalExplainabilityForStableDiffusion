import numpy as np
from typing import List
from transformers import CLIPModel, CLIPProcessor
from diffusers import LCMScheduler, AutoPipelineForText2Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


from patched_call import patched_call
from patched_call_lcm import patched_call_lcm
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
#model_id = "runwayml/stable-diffusion-v1-5"
model_id = "Lykon/dreamshaper-7"
pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype, variant="fp16").to(device)

adapter_id = "latent-consistency/lcm-lora-sdv1-5"
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

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


prompt = "A photograph of a bowl full of berries"
prompt = "The artwork portrays a cute goblin girl with a devious expression and a playful air. She has delicate, sharp features and large, alert eyes that seem to radiate curiosity and intelligence. Her ears are large and pointed, a signature feature of goblins. Her hair is a wild, dark mess, adding to her wild and untamed nature."
prompt = "A tense, dramatic scene from the television series Peaky Blinders. The air is thick with anticipation as two rival gangs prepare to fight in the dark woods. A cabin stands in the background, providing a looming presence and the potential for escape or refuge. The faces of the gang members are covered in shadows, their features sharp and defined. The image is rendered in black and white, providing a stark and timeless atmosphere. The focus is on the characters' faces, their expressions ranging from determination to fear, all captured in detailed, symmetrical profiles. The camera is positioned to give a medium-long shot, capturing the full breadth of the scene while maintaining a cinematic and epic feel"
prompt = "apple photograph black moldy discusting worm-filled rotten "

def hpsv2_loss_against_null(images, _):
    batch_size = images.shape[0]
    # processes image using patched clip processor
    clip_image_inputs = clip_image_processor.preprocess(images, clip_processor.image_processor)

    clip_text_inputs = clip_processor(text=[""]*batch_size, return_tensors="pt", padding=True, truncation=True)
    clip_text_inputs = {k:v.to(device) for k,v in clip_text_inputs.items()}

    clip_outputs = clip_model(pixel_values=clip_image_inputs, **clip_text_inputs)

    loss = (clip_outputs.text_embeds * clip_outputs.image_embeds).sum(-1).mean()
    return loss

def hpsv2_loss(images, prompts):
    # processes image using patched clip processor
    clip_image_inputs = clip_image_processor.preprocess(images, clip_processor.image_processor)

    clip_text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    clip_text_inputs = {k:v.to(device) for k,v in clip_text_inputs.items()}

    clip_outputs = clip_model(pixel_values=clip_image_inputs, **clip_text_inputs)

    loss = (clip_outputs.text_embeds * clip_outputs.image_embeds).sum(-1).mean()
    return loss

def call_pipe_and_get_prompt_grads(pipe, prompts, loss_fn, pipe_kwargs):
    out = patched_call(
        self=pipe, 
        prompt=prompts, 
        output_type='pt', 
        num_inference_steps = 4, 
        **pipe_kwargs,
    )


    # Get the global variable we saved 
    # the 0th hidden state is the text input embedding
    hidden_states = clip_text_forward_patch.global_hidden_states[0]
    clip_text_forward_patch.global_hidden_states = []

    loss = loss_fn(out.images, prompts)
    loss.backward()

    return hidden_states.grad, hidden_states.detach(), out.images


def integrated_grads(pipe, prompts, loss_fn, pipe_kwargs={}, ):

    gs_start = 0.000001

    gs_end = 1.7

    ig_steps = 2


    grads = 0.0
    for guidance_scale in tqdm(np.linspace(gs_start, gs_end, ig_steps)):
        pipe_kwargs['guidance_scale'] = guidance_scale
        pipe_kwargs['generator'] = torch.Generator().manual_seed(42)

        new_grads, hidden_states, images = call_pipe_and_get_prompt_grads(pipe, prompts, loss_fn, pipe_kwargs)
        grads = grads + new_grads

    # takes mean
    grads = grads / ig_steps

    return grads, hidden_states, images



grads, hidden_states, images = integrated_grads(pipe, [prompt], hpsv2_loss_against_null)

image = images[0]

plt.imshow(transforms.ToPILImage()(image), interpolation="bicubic")
plt.savefig('./image.jpg')

input_ids = pipe.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True)['input_ids'][0]

mask_not_padding = input_ids != pipe.tokenizer.pad_token_id

hidden_states_grad = grads[0, mask_not_padding,:]
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






# Convert the PyTorch tensor to a PIL image
original_image = transforms.ToPILImage()(image).convert("RGBA")

# Prepare text data from dot_product
words = [pipe.tokenizer.decode(token_id) for token_id in input_ids]
attributions = [attribution.item() * 100 for attribution in dot_product]

# Create a new image for heatmap text
heatmap_height = 200
heatmap_image = Image.new("RGBA", (original_image.width, heatmap_height), (255, 255, 255, 0))
draw = ImageDraw.Draw(heatmap_image)

# Use a specific TrueType font with increased size
font_size = 16
font_path = 'OpenSans-Regular.ttf'
font = ImageFont.truetype(font_path, font_size)

# Initialize the position for drawing text
x_pos, y_pos = 0, 0
line_height = font.getlength('Ag') + 5  # Height of a line of text

def will_text_exceed_width(text, x_pos, image_width, font):
    return x_pos + font.getlength(text) > image_width

min_attribution = min(attributions)
max_attribution = max(attributions)

# Function to normalize and scale the attributions
def scale_attribution(value, min_val, max_val, new_min, new_max):
    normalized = (value - min_val) / (max_val - min_val)
    return normalized * (new_max - new_min) + new_min

# Scale the attributions to the new range of 0.2 to 1.0
scaled_attributions = [scale_attribution(attr, min_attribution, max_attribution, 0.2, 1.0) for attr in attributions]

# Create heatmap based on scaled attributions
for word, scaled_attr in zip(words[1:], scaled_attributions[1:]): # Skip "<startoftext>"
    if will_text_exceed_width(word, x_pos, heatmap_image.width, font):
        # Move to next line if the word exceeds the width
        x_pos = 0
        y_pos += line_height

    color = mcolors.to_hex(plt.cm.Reds(scaled_attr)) 
    draw.text((x_pos, y_pos), word, fill=color, font=font)
    x_pos += font.getlength(word) + 5  # Adjust x position for next word

# Combine original image and heatmap
combined_image = Image.new("RGBA", (original_image.width, original_image.height + heatmap_height))
combined_image.paste(original_image, (0, 0))
combined_image.paste(heatmap_image, (0, original_image.height))

combined_image_rgb = combined_image.convert("RGB")
combined_image_rgb.save('./modified_image.jpg', 'JPEG')
