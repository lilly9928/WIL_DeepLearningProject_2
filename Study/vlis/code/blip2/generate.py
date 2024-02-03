from typing import Optional, List, Dict, Any, Tuple
from pprint import pprint

import torch
from PIL import Image

from load import Model
from score import Scorer


vm_template_v1 = 'Question: {} Answer:'
vm_template_v2 = '{}'

lm_template_v1 = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
    {}

### Input:
    {}

### Response:
"""


lm_template_v2 = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
    {}

### Response:
"""


class GenerationModel(Model, Scorer):
    def encode_and_generate(
        self,
        image_good,
        images_bad: Optional[list] = None,
        vm_prompt: str = '',
        lm_prompt: str = '',
        device: str = 'cuda:0',
        **generation_args
    ):
        if images_bad is None:
            size = image_good.size
            images_bad = [
                Image.new('RGB', size, (0, 0, 0)),  # black
                Image.new('RGB', size, (255, 255, 255))   # white
            ]
        imgs = self.processor(text=vm_prompt, images=[image_good, *images_bad], return_tensors='pt').to(torch.float16)
        txts = self.tokenizer(lm_prompt, padding=True, return_tensors='pt')
        data = dict(
            pixel_values_good=imgs['pixel_values'][:1],
            pixel_values_bad=imgs['pixel_values'][1:].unsqueeze(1),
            input_ids_vm=imgs['input_ids'],
            attention_mask_vm=imgs['attention_mask'],
            input_ids_lm=txts['input_ids'],
            attention_mask_lm=txts['attention_mask'],
        )
        data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
        outputs = self.generate(**data, **generation_args,
                                pad_token_id=self.processor.tokenizer.pad_token_id,
                                eos_token_id=self.processor.tokenizer.eos_token_id,
                                early_stopping=True)
        generated = self.processor.batch_decode(outputs, skip_special_tokens=True)
        generated = generated[0].strip()
        return generated

    def run(self, img, instruction, inputs = None, device: str = 'cuda:0',
            **generation_args):
        with torch.no_grad():
            vm_prompt = format_vm_prompt(instruction)
            lm_prompt = format_lm_prompt(instruction, inputs)
            out = self.encode_and_generate(
                image_good=img,
                vm_prompt=vm_prompt,
                lm_prompt=lm_prompt,
                device=device,
                min_new_tokens=2,
                **generation_args
            )
        return out


def format_vm_prompt(instruction):
    instruction = instruction.strip()
    if not (instruction.endswith('.') or instruction.endswith('?')):
        instruction = f'{instruction}.'
    return vm_template_v2.strip().format(instruction)


def format_lm_prompt(instruction, inputs):
    if inputs:
        x = lm_template_v1.strip()
        x = x.format(instruction, inputs)
    else:
        x = lm_template_v2.strip()
        x = x.format(instruction)
    x = x + '\n'
    return x


if __name__ == '__main__':
    instruction = 'Which one of these animals is native to north america?'
    inputs = None
    print('lm')
    pprint(format_lm_prompt(instruction, inputs))
    print('vm')
    pprint(format_vm_prompt(instruction))
