import os
import numpy as np
from typing import List
from transformers import CLIPModel, CLIPProcessor
from diffusers import LCMScheduler, AutoPipelineForText2Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


from patched_call import patched_call_sd
from patched_call_lcm import patched_call_lcm
from patched_call_sdxl import patched_call_sdxl
import clip_image_processor
import clip_text_forward_patch

from llava_reward import LlavaRewardQA

device = "cuda"
dtype = torch.float16
# device='cpu'
# dtype = torch.float32

# Set seed for deterministic outcome
seed = 16
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
generator = torch.Generator(device=device).manual_seed(seed)

# loads stable diffusion pipe
model_id = "Lykon/dreamshaper-7"
#model_id = "stabilityai/sdxl-turbo"
#model_id = "runwayml/stable-diffusion-v1-5"
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
        pipe.text_encoder.text_model, *args,
        **kwargs
    )
)


def call_pipe_and_get_prompt_grads(pipe, prompt:str, seed:int, pipe_kwargs={}, 
    questions=["What is the quality of this image?", "What is the main subject of this image?", "What things are in this image?"],
    question_options=["A. Terrible\nB. Poor\nC. Ok\nD. Good\nE. Excellent", "A. A cripsy red apple\nB. An apple logo\nC. A apple that is black and decomposing\nD. A decomposing apple with a worm in it", "A. Apple in the tree\nB. Apple on a branch\n C. Apple on a table\nD. A worm eating an apple"],
    answers=["E", "D", "D"],
   n_images_per_prompt=1
       ):


    accum_steps = len(questions)
    # really hacky manual gradient accumulation
    grad = 0.0
    ret_hidden_states = None
    ret_image = None
    pipe_kwargs["generator"] = torch.Generator().manual_seed(seed)

    if pipe_kwargs.get("guidance_scale") is None:
        pipe_kwargs["guidance_scale"] = 1.7

    for question, options, answer in zip(
        questions, question_options, answers
    ):
        for _ in range(n_images_per_prompt):
            out = patched_call_sd(
                self=pipe,
                prompt=prompt,
                output_type="pt",
                num_inference_steps=8,
                **pipe_kwargs,
            )
            # batch size of one for loss
            image = out.images[0].unsqueeze(0)

            # Get the global variable we saved
            # the 0th hidden state is the text input embedding
            hidden_states = clip_text_forward_patch.global_hidden_states[0]
            clip_text_forward_patch.global_hidden_states = []

            loss, reward = llava_reward_qa(image, question, options, answer)
            print("got loss rfn", loss)
            # scales the loss so that it is bigger and doesn't underflow
            scale = 1e2
            ((scale * loss) / (accum_steps * n_images_per_prompt)).backward()

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
    questions=["What is the quality of this image?", "What is the main subject of this image?", "What things are in this image?"],
    question_options=["A. Terrible\nB. Poor\nC. Ok\nD. Good\nE. Excellent", "A. A cripsy red apple\nB. An apple logo\nC. A apple that is black and decomposing\nD. A decomposing apple with a worm in it", "A. Apple in the tree\nB. Apple on a branch\n C. Apple on a table\nD. A worm eating an apple"],
    answers=["E", "D", "D"],
):
    gs_start = 0.000001

    gs_end = 1.5

    ig_steps = 10

    grads = 0.0
    for guidance_scale in tqdm(np.linspace(gs_start, gs_end, ig_steps)):
        pipe_kwargs["guidance_scale"] = guidance_scale
        new_grads, hidden_states, images = call_pipe_and_get_prompt_grads(
            pipe, prompts, seed, pipe_kwargs,questions, question_options, answers,
        )
        grads = grads + new_grads

    # takes mean
    grads = grads / ig_steps

    return grads, hidden_states, images


import labels_qa_handmade
import average_precision

# all mean average precisions
all_maps_at_10 = []
all_maps_at_5 = []
# mean average_precision @ inf
mmap = []

rmap_at_10 = []
# randomized map at inf
rmap = []

os.makedirs("out/", exist_ok=True)

for i, (prompt,question,options, answer,token) in enumerate(zip(labels_qa_handmade.prompts,
                  labels_qa_handmade.questions,
                  labels_qa_handmade.options,
                  labels_qa_handmade.answers,
                  labels_qa_handmade.token)):

    grads, hidden_states, images = call_pipe_and_get_prompt_grads(pipe, prompt, random.randint(0, 10000), {"guidance_scale":0.0}, [question], [options], [answer], n_images_per_prompt=8)
    #grads, hidden_states, images = integrated_grads(pipe, prompt, random.randint(0, 10000), {}, [question], [options], [answer])

    image = images[0]

    input_ids = pipe.tokenizer(
        prompt, return_tensors="pt", padding="max_length", truncation=True
    )["input_ids"][0]

    mask_not_padding = input_ids != pipe.tokenizer.pad_token_id

    hidden_states_grad = grads[0, mask_not_padding, :]
    hidden_states = hidden_states[0, mask_not_padding, :]

    # Dot product the gradients with the embeddings
    # Sum the element wise multiplication across the last dimension
    dot_product = (hidden_states * hidden_states_grad).sum(-1)

    ranked_word_indices = dot_product.cpu().sort(descending=True).indices

    dot_product = dot_product - dot_product.min()
    dot_product = dot_product / dot_product.sum()

    target_tokens = [token]

    all_maps_at_10.append(average_precision.apk(target_tokens, ranked_word_indices.tolist(), k=10))
    all_maps_at_5.append(average_precision.apk(target_tokens, ranked_word_indices.tolist(), k=5))
    mmap.append(average_precision.apk(target_tokens, ranked_word_indices.tolist(), k=len(ranked_word_indices)))

    print("mAP: ", mmap[-1])
    print("perfect precision:",average_precision.apk([ranked_word_indices.tolist()[0]], ranked_word_indices.tolist(), k=len(ranked_word_indices)))

    _rmaps = []
    _rmaps_at_10 = []
    for _ in range(25):
        ranked_word_indices_random = np.random.permutation(len(ranked_word_indices))
        _rmaps.append(average_precision.apk(target_tokens, ranked_word_indices_random.tolist(), k=len(ranked_word_indices)))
        _rmaps_at_10.append(average_precision.apk(target_tokens, ranked_word_indices_random.tolist(), k=10))
    rmap.append(np.array(_rmaps).mean())
    rmap_at_10.append(np.array(_rmaps_at_10).mean())


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
    font_size = 24
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
    combined_image_rgb.save(f"./out/{i:03}modified_image.jpg", "JPEG")

    print("Average mAP @ 5", np.array(all_maps_at_5).mean())
    print("Average mAP @ 10", np.array(all_maps_at_10).mean())
    print("Average mAP @ inf", np.array(mmap).mean())
    print("Average random mAP @ inf", np.array(rmap).mean())

print("Average mAP @ 5", np.array(all_maps_at_5).mean())
print("Average mAP @ 10", np.array(all_maps_at_10).mean())
print("Average mAP @ inf", np.array(mmap).mean())
print("Average mAP @ inf 95% CI", average_precision.mean_confidence_interval(mmap))
print("Average random mAP @ inf", np.array(rmap).mean())
print("Average random mAP @ inf 95% CI", average_precision.mean_confidence_interval(rmap))
print("Average random mAP @ 10", np.array(rmap_at_10).mean())

import scipy.stats

x = ["MCQA","random"]
y = [np.array(mmap).mean(), np.array(rmap).mean()]

def n5_ci(arr):
    se = scipy.stats.sem(arr)
    h = se * scipy.stats.t.ppf((1 + 0.95) / 2., len(arr)-1)
    return h
c = [n5_ci(mmap), 0]

plt.close()
plt.figure(figsize=(3,4))
plt.bar(x, y)
plt.errorbar(x, y, yerr=c, fmt="o", color="r")
plt.ylabel("mAP@inf")
plt.savefig("./mAP@inf figure.jpg",dpi=300)

plt.close()
plt.figure(figsize=(3,4))
y = [np.array(all_maps_at_10).mean(), np.array(rmap_at_10).mean()]
plt.bar(x, y)
plt.errorbar(x, y, yerr=c, fmt="o", color="r")
plt.ylabel("mAP@10")
plt.savefig("./mAP@10 figure.jpg",dpi=300)

print("ttest @ inf",scipy.stats.ttest_1samp(mmap, np.array(rmap).mean(), alternative='greater'))
print("ttest @ 10",scipy.stats.ttest_1samp(all_maps_at_10, np.array(rmap_at_10).mean(), alternative='greater'))
