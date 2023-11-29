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

from llava_reward import LlavaRewardQA

device = "cuda"
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
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id, torch_dtype=dtype, variant="fp16"
).to(device)

adapter_id = "latent-consistency/lcm-lora-sdv1-5"
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

## loads Human Preference Score v2
# clip_model = CLIPModel.from_pretrained("adams-story/HPSv2-hf", torch_dtype=dtype).to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") # uses the same exact vanilla clip processor

# loads LlavaRewardQA
llava_reward_qa = LlavaRewardQA(device=device)

pipe.safety_checker = None

pipe = pipe.to(device)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

# Use our patched clip_text_forward method
pipe.text_encoder.text_model.forward = (
    lambda *args, **kwargs: clip_text_forward_patch.forward(
        pipe.text_encoder.text_model, *args, **kwargs
    )
)


def call_pipe_and_get_prompt_grads(pipe, prompt:str, seed:int, pipe_kwargs={}, 
    questions=["What is the quality of this image?", "What is the main subject of this image?", "What things are in this image?"],
    question_options=["A. Terrible\nB. Poor\nC. Ok\nD. Good\nE. Excellent", "A. A cripsy red apple\nB. An apple logo\nC. A apple that is black and decomposing\nD. A decomposing apple with a worm in it", "A. Apple in the tree\nB. Apple on a branch\n C. Apple on a table\nD. A worm eating an apple"],
    answers=["E", "D", "D"],
       ):


    accum_steps = len(questions)
    # really hacky manual gradient accumulation
    grad = 0.0
    ret_hidden_states = None
    ret_image = None
    for question, options, answer in zip(
        questions, question_options, answers
    ):
        if pipe_kwargs.get("guidance_scale") is None:
            pipe_kwargs["guidance_scale"] = 1.7
        pipe_kwargs["generator"] = torch.Generator().manual_seed(seed)
        out = patched_call(
            self=pipe,
            prompt=prompt,
            output_type="pt",
            num_inference_steps=8,
            **pipe_kwargs,
        )
        image = out.images[0].unsqueeze(0)

        # Get the global variable we saved
        # the 0th hidden state is the text input embedding
        hidden_states = clip_text_forward_patch.global_hidden_states[0]
        clip_text_forward_patch.global_hidden_states = []

        loss, reward = llava_reward_qa(image, question, options, answer)
        print("got loss rfn", loss)
        (loss /accum_steps).backward()

        grad += hidden_states.grad

        ret_hidden_states = hidden_states.detach()
        ret_image = image.detach()

        del hidden_states
        del loss
        del reward
        torch.cuda.empty_cache()
        

    return grad, ret_hidden_states, ret_image


def integrated_grads(
    pipe,
    prompts,
    seed=42,
    pipe_kwargs={},
):
    gs_start = 0.000001

    gs_end = 1.7

    ig_steps = 5

    grads = 0.0
    for guidance_scale in tqdm(np.linspace(gs_start, gs_end, ig_steps)):
        pipe_kwargs["guidance_scale"] = guidance_scale
        new_grads, hidden_states, images = call_pipe_and_get_prompt_grads(
            pipe, prompts, seed, pipe_kwargs
        )
        grads = grads + new_grads

    # takes mean
    grads = grads / ig_steps

    return grads, hidden_states, images


prompt = "A photograph of a bowl full of berries"
prompt = "The artwork portrays a cute goblin girl with a devious expression and a playful air. She has delicate, sharp features and large, alert eyes that seem to radiate curiosity and intelligence. Her ears are large and pointed, a signature feature of goblins. Her hair is a wild, dark mess, adding to her wild and untamed nature."
prompt = "A tense, dramatic scene from the television series Peaky Blinders. The air is thick with anticipation as two rival gangs prepare to fight in the dark woods. A cabin stands in the background, providing a looming presence and the potential for escape or refuge. The faces of the gang members are covered in shadows, their features sharp and defined. The image is rendered in black and white, providing a stark and timeless atmosphere. The focus is on the characters' faces, their expressions ranging from determination to fear, all captured in detailed, symmetrical profiles. The camera is positioned to give a medium-long shot, capturing the full breadth of the scene while maintaining a cinematic and epic feel"
prompt = "apple photograph black moldy discusting eaten by worms and rotten "

while True:
    prompt = input("image prompt:").strip()
    grads, hidden_states, images = integrated_grads(pipe, prompt, 42)

    image = images[0]

    plt.imshow(transforms.ToPILImage()(image), interpolation="bicubic")
    plt.savefig("./image.jpg")

    input_ids = pipe.tokenizer(
        prompt, return_tensors="pt", padding="max_length", truncation=True
    )["input_ids"][0]

    mask_not_padding = input_ids != pipe.tokenizer.pad_token_id

    hidden_states_grad = grads[0, mask_not_padding, :]
    hidden_states = hidden_states[0, mask_not_padding, :]

    # Dot product the gradients with the embeddings
    # Sum the element wise multiplication across the last dimension
    dot_product = (hidden_states * hidden_states_grad).sum(-1)

    dot_product = dot_product - dot_product.min()
    dot_product = dot_product / dot_product.sum()


    for attribution, token_id in zip(dot_product, input_ids):
        print(
            f'token: "{pipe.tokenizer.decode(token_id):<15}"   attribution: {attribution.item() * 100:6.1f}%'
        )
    # Print the dot product
    print("Dot product of gradients with embeddings:", dot_product)


    # Convert the PyTorch tensor to a PIL image
    original_image = transforms.ToPILImage()(image).convert("RGBA")

    # Prepare text data from dot_product
    words = [pipe.tokenizer.decode(token_id) for token_id in input_ids]
    attributions = [attribution.item() * 100 for attribution in dot_product]

    # Create a new image for heatmap text
    heatmap_height = 200
    heatmap_image = Image.new(
        "RGBA", (original_image.width, heatmap_height), (255, 255, 255, 0)
    )
    draw = ImageDraw.Draw(heatmap_image)

    # Use a specific TrueType font with increased size
    font_size = 16
    font_path = "OpenSans-Regular.ttf"
    font = ImageFont.truetype(font_path, font_size)

    # Initialize the position for drawing text
    x_pos, y_pos = 0, 0
    line_height = font.getlength("Ag") + 5  # Height of a line of text


    def will_text_exceed_width(text, x_pos, image_width, font):
        return x_pos + font.getlength(text) > image_width


    min_attribution = min(attributions)
    max_attribution = max(attributions)


    # Function to normalize and scale the attributions
    def scale_attribution(value, min_val, max_val, new_min, new_max):
        normalized = (value - min_val) / (max_val - min_val)
        return normalized * (new_max - new_min) + new_min


    # Scale the attributions to the new range of 0.2 to 1.0
    scaled_attributions = [
        scale_attribution(attr, min_attribution, max_attribution, 0.2, 1.0)
        for attr in attributions
    ]

    # Create heatmap based on scaled attributions
    for word, scaled_attr in zip(
        words[1:], scaled_attributions[1:]
    ):  # Skip "<startoftext>"
        if will_text_exceed_width(word, x_pos, heatmap_image.width, font):
            # Move to next line if the word exceeds the width
            x_pos = 0
            y_pos += line_height

        color = mcolors.to_hex(plt.cm.Reds(scaled_attr))
        draw.text((x_pos, y_pos), word, fill=color, font=font)
        x_pos += font.getlength(word) + 5  # Adjust x position for next word

    # Combine original image and heatmap
    combined_image = Image.new(
        "RGBA", (original_image.width, original_image.height + heatmap_height)
    )
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(heatmap_image, (0, original_image.height))

    combined_image_rgb = combined_image.convert("RGB")
    combined_image_rgb.save("./modified_image.jpg", "JPEG")
