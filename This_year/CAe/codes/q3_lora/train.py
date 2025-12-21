"""
LoRA training utilities (skeleton)
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_model_and_tokenizer(model_name='meta-llama/Llama-3.2-1B'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    return model, tokenizer

def apply_lora(model, r=8, alpha=32, target_modules=None, dropout=0.1):
    if target_modules is None:
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    config = LoraConfig(r=r, lora_alpha=alpha, target_modules=target_modules, lora_dropout=dropout, bias='none')
    model = get_peft_model(model, config)
    return model
