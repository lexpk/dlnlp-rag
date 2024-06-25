from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import torch


class Generator:
    def __init__(self):
        pass
    
    def __call__(self, prompt, max_new_tokens=100, seed=42):
        torch.manual_seed(seed)        
        output = self.pipeline(prompt, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)[0]['generated_text']
        return output

    def get_tokenizer(self):
        return self.tokenizer


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
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        self.tokenizer = tokenizer
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


class Llama3_8b(Generator):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        self.tokenizer = tokenizer
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

class Llama2_4b(Generator):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        model.config.pad_token_id = model.generation_config.eos_token_id
        self.tokenizer = tokenizer
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

class Gpt2(Generator):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2",
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        self.tokenizer = tokenizer
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

class Phi15_1b(Generator):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-1_5",
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        self.tokenizer = tokenizer
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
