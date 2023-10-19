from diffusers import StableDiffusionPipeline
import torch
from torch.utils.checkpoint import checkpoint

from patched_call import patched_call

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.safety_checker=None


pipe = pipe.to("cuda")
pipe.text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)
#pipe.vae.requires_grad_(True)
#pipe.unet.requires_grad_(True)
#pipe.text_encoder.requires_grad_(True)



prompt = "a photo of an astronaut riding a horse on mars"

#out = checkpoint(lambda p,pr:patched_call(p, pr, output_type='pt', num_inference_steps = 5,), pipe, prompt, debug=True, use_reentrant=False)
out = patched_call(pipe, prompt, output_type='pt', num_inference_steps = 5,)
image = out.images[0]

image.mean().backward()
    
print("grad sum", pipe.text_encoder.text_model.embeddings.token_embedding.weight.grad.sum())
