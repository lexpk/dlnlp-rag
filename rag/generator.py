from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import torch


class Generator:
    def __init__(self):
        pass
    
    def __call__(self, prompt, max_new_tokens=100):
        output = self.pipeline(prompt, max_new_tokens=max_new_tokens)[0]['generated_text']
        return output


class Llama3_70b_4bit(Generator):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


class Llama3_8b(Generator):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
