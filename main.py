from diffusers import StableDiffusionPipeline
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

from patched_call import patched_call

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

pipe.vae.enable_gradient_checkpointing()
pipe.unet.enable_gradient_checkpointing()
pipe.text_encoder._set_gradient_checkpointing(True)

pipe.safety_checker=None


pipe = pipe.to("cuda")
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

prompt = "a photo of an apple"

#out = checkpoint(lambda p,pr:patched_call(p, pr, output_type='pt', num_inference_steps = 5,), pipe, prompt, debug=True, use_reentrant=False)
out = patched_call(pipe, prompt, output_type='pt', num_inference_steps = 20,)
image = out.images[0]

loss = image.sum()

loss.backward()

print("grad sum", pipe.text_encoder.text_model.embeddings.token_embedding.weight.grad.sum())

input_ids = pipe.tokenizer(prompt, return_tensors="pt",)['input_ids'].to(pipe.device)[0]

# shape: (input_ids, 768)
input_token_embedding_grad = pipe.text_encoder.text_model.embeddings.token_embedding.weight.grad[input_ids]


plt.imshow(transforms.ToPILImage()(image), interpolation="bicubic")
plt.savefig("image.jpg")
