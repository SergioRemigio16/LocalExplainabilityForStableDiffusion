from torch.nn import functional as F
from typing import Dict, List
from llava.model.builder import load_pretrained_model
from torchvision.transforms import InterpolationMode
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
import torch
from llava import LlavaLlamaForCausalLM
import torchvision
import llava.mm_utils as llava_utils
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    process_images,
)
from llava.conversation import conv_templates as llava_conv_templates


# exactly the same as the original https://github.com/haotian-liu/LLaVA/blob/785f766fcddc86ffeaa62cd51cf7834a11c04e6d/llava/model/multimodal_encoder/clip_encoder.py#L40C10-L40C10
# but with the @no_grad() removed
def _clip_vision_tower_forward(self_vision_tower, images):
    if type(images) is list:
        image_features = []
        for image in images:
            image_forward_out = self_vision_tower.vision_tower(
                image.to(
                    device=self_vision_tower.device, dtype=self_vision_tower.dtype
                ).unsqueeze(0),
                output_hidden_states=True,
            )
            image_feature = self_vision_tower.feature_select(image_forward_out).to(
                image.dtype
            )
            image_features.append(image_feature)
    else:
        image_forward_outs = self_vision_tower.vision_tower(
            images.to(device=self_vision_tower.device, dtype=self_vision_tower.dtype),
            output_hidden_states=True,
        )
        image_features = self_vision_tower.feature_select(image_forward_outs).to(
            images.dtype
        )

    return image_features


class LlavaRewardQA:
    def __init__(
        self,
        inference_dtype=torch.float16,
        device="cuda",
        model_path="teowu/llava_v1.5_7b_qinstruct_preview_v0.1",
        max_seq_len: int = 256,
        torch_compile=False,
    ):
        print("LOADING LLAVA MODEL")

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_path.split("/")[-1],
            load_4bit=True,
        )

        model = model.eval()

        model.gradient_checkpointing_enable()
        model.model.gradient_checkpointing_enable()
 

        self.model = model
        self.image_processor = image_processor

        for mod in model.modules():
            mod.requires_grad_(False)

        model.gradient_checkpointing_enable()
        model.model.gradient_checkpointing_enable()

        # needs to monkey patch the vision tower so it doesn't have the @no_grad decorator
        model.model.vision_tower.forward = lambda images: _clip_vision_tower_forward(
            model.model.vision_tower, images
        )

        if torch_compile:
            model = torch.compile(model)

        print("DONE LOADING LLAVA MODEL")

        model_name = llava_utils.get_model_name_from_path(model_path)
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        self.max_seq_len = min(max_seq_len, context_len)

        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.tokenizer = tokenizer
        self.conv_template = llava_conv_templates[conv_mode].copy()
        self.device = device
        self.dtype = inference_dtype

        self.collator = DataCollatorForSeq2Seq(
            self.tokenizer, None, padding=True, pad_to_multiple_of=64
        )

    def _process_image_pixels(self, x: torch.Tensor):
        """
        Does everything that the image_processor (CLIPImageProcessor)
        does, but using native pytorch differentiable operations
        """
        # x = ((x / 2) + 0.5).clamp(0, 1)
        to_h, to_w = (
            self.image_processor.crop_size["height"],
            self.image_processor.crop_size["width"],
        )
        x = F.interpolate(x, (to_h, to_w), antialias=False, mode="nearest")

        # normalizes
        mean = self.image_processor.image_mean
        std = self.image_processor.image_std

        x = torchvision.transforms.Normalize(mean=mean, std=std)(x)

        return x

    def __call__(self, pixel_values: torch.Tensor, question, options, answer):
        """
        pixel_values: batch of pixel values
        batched_prompt_d: batch of prompt dicts, where each dict has a prompt_qa_questions, prompt_qa_answers and prompt_qa_options

        prompt_qa_questions is a list of string questions

        prompt_qa_answers is a list of string len 0 answers

        prompt_qa_options is a list of the string options

        Returns a single scalar (loss,reward) for the whole batch
        """
        pixel_values = (
            self._process_image_pixels(pixel_values).to(self.device).to(self.dtype)
        )

        inp = f"{question}\nAnswer with the option’s letter from the given choices directly.\n{options}"
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).to(self.device).unsqueeze(0)

        answer_id = self.tokenizer(
            answer,
            add_special_tokens=False,
        )[
            "input_ids"
        ][0]

        outputs = self.model(images=pixel_values, return_dict=True, input_ids=input_ids)

        # we want the correct answer token to be really positive
        # and we want the other answer tokens to be really negative
        loss = -outputs.logits[0, -1, answer_id]

        # just the normal llm loss
        #loss = torch.nn.functional.cross_entropy(outputs.logits[0,-1].unsqueeze(0), torch.LongTensor([answer_id]).to(self.device))

        return loss, -loss
