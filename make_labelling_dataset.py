import datasets
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5"
        )
tokenizer = pipeline.tokenizer


diffusiondb = datasets.load_dataset("poloclub/diffusiondb", )

diffusiondb = diffusiondb.shuffle(42)

for i in range(100):
    p = diffusiondb['train'][i]['prompt']
    input_ids = tokenizer(p)['input_ids']
    dec_p = [tokenizer.decode(x) for x in input_ids]
    print(f"{i}: \n\tprompt: {p}\ntokens: ", end="")
    for j, t in enumerate(dec_p):
        print(f"{j} '{t}', ", end = "")
    print()


