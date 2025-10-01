from transformers import AutoModelForCausalLM, AutoTokenizer

class SmolLM:
    checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"
    device = "cpu"
    tokenizer = None
    model = None
    
    @staticmethod
    def load():
        SmolLM.tokenizer = AutoTokenizer.from_pretrained(SmolLM.checkpoint)
        SmolLM.model = AutoModelForCausalLM.from_pretrained(SmolLM.checkpoint).to(SmolLM.device)
    
    @staticmethod
    def infer(prompt: str, max_new_tokens=100, temperature=0.2, top_p=0.9, do_sample=True):
        if SmolLM.tokenizer is None or SmolLM.model is None:
            raise ValueError("Model and tokenizer must be loaded before inference. Call SmolLM.load() first.")
        
        messages = [{"role": "user", "content": prompt}]
        
        input_text = SmolLM.tokenizer.apply_chat_template(messages, tokenize=False)
        
        inputs = SmolLM.tokenizer.encode(input_text, return_tensors="pt").to(SmolLM.device)
        outputs = SmolLM.model.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample)
        return SmolLM.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).replace("assistant\n", "").strip()