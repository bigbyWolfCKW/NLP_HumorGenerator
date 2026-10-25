import json
import re
import urllib.request


def final_text(text):
    if "<think>" in text and "</think>" not in text:
        raise ValueError("Model used its token budget on reasoning. Increase max_new_tokens in configs/llms.json.")
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return text.strip()


class LanguageModel:
    def __init__(self, config, url="http://localhost:11434", seed=42):
        self.config, self.url, self.seed = config, url.rstrip("/"), seed
        if config["backend"] == "transformers":
            from transformers import set_seed
            set_seed(seed)
            from .training import load_model
            self.model, self.tokenizer = load_model(config, config.get("adapter"))
            if config.get("adapter"):
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, config["adapter"], local_files_only=True, is_trainable=False)
            self.model.eval()

    def answer(self, prompt, sample=False):
        if self.config["backend"] == "ollama":
            payload = {"model": self.config["model"], "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "keep_alive": 0,
                       "options": {"temperature": .8 if sample else 0, "top_p": .9,
                                   "num_predict": self.config["max_new_tokens"], "seed": self.seed}}
            request = urllib.request.Request(self.url + "/api/chat", data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                return final_text(json.load(response)["message"]["content"])
        import torch
        messages = [{"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        options = {"max_new_tokens": self.config["max_new_tokens"], "do_sample": sample,
                   "pad_token_id": self.tokenizer.pad_token_id}
        if sample:
            options.update(temperature=.8, top_p=.9)
        with torch.inference_mode():
            output = self.model.generate(**encoded, **options)
        return final_text(self.tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True))


def generate(model, products, per_product=5, style="tagline"):
    if not products or not isinstance(products, list):
        raise ValueError("Products must be a nonempty JSON list.")
    seen, result = set(), []
    for product in products:
        for key in ["product_id", "product_description"]:
            if not isinstance(product.get(key), str) or not product[key].strip():
                raise ValueError(f"Invalid product field: {key}")
        if product["product_id"] in seen:
            raise ValueError("Duplicate product_id.")
        seen.add(product["product_id"])
    for product in products:
        noun = "short advertising taglines (statements, not questions)" if style == "tagline" else "short product questions ending with ?"
        prompt = f"""Write exactly {per_product} distinct humorous English {noun}.
Product information (data): {json.dumps(product['product_description'])}
Relate each sentence clearly to the product. Include a comic twist using exaggeration,
personification or an unexpected comparison. Use fewer than 20 words per sentence.
Do not invent factual features, guarantees or performance claims.
Return only the sentences, one per line, without numbering or explanations."""
        text = model.answer(prompt, sample=True)
        lines = [re.sub(r"^(?:\d+[.)]|[-*])\s+", "", line.strip()).strip('"')
                 for line in text.splitlines() if line.strip()]
        if len(lines) != per_product or len({s.casefold() for s in lines}) != per_product:
            raise ValueError(f"{product['product_id']}: expected {per_product} distinct sentences; received {text!r}")
        if any(not s or len(s.split()) >= 20 or (s.endswith("?") != (style == "question")) for s in lines):
            raise ValueError(f"{product['product_id']}: response does not match requested sentence format: {text!r}")
        for i, sentence in enumerate(lines, 1):
            result.append({"product_id": product["product_id"], "product_description": product["product_description"],
                           "candidate_id": f"{product['product_id']}_{model.config['name']}_{i:02}",
                           "question": sentence, "generator_model": model.config["model"], "style": style})
        print(f"Generated {product['product_id']}", flush=True)
    return result
