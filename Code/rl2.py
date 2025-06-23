from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from web_agent_site.new_agent import SimpleAgent
from trl import GRPOConfig
from verifiers.trainers.rl3 import GRPOEnvTrainer
from bitsandbytes.optim import AdamW8bit
from torch.utils.data import Dataset
import torch
import re

class WebShopGRPOEnv:
    def __init__(self, agent, tokenizer, model):
        self.agent = agent
        self.tokenizer = tokenizer
        self.model = model

    def model_chat(self, prompt):
        prompt_text = self.tokenizer.apply_chat_template(prompt, tokenize=False) + "<|im_start|>assistant\n"
        #print("Prompt text:\n", prompt_text)
        encoding = self.tokenizer(prompt_text,return_tensors="pt",padding=False, add_special_tokens=False).to(self.model.device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
        generated_ids = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        #print("Decoded response:", repr(response))
        return response

    def generate(self, instruction):
    
        # instruction = self.agent.start_session()
        print("\n Instruction:", instruction)
    
        prompt_query = [
            {"role": "system", "content": "You are a helpful assistant that writes concise e-commerce search queries."},
            {"role": "user", "content": (
                f"Instruction: {instruction}\n\n"
                "Now generate only the search query. Do not include any extra explanation or formatting. "
                "Just write the query as a single line."
            )}
        ]

        query = self.model_chat(prompt_query).strip()
        query = query.replace("|", "").strip()

        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1].strip()

        print("Query:", query)

        if not query:
            print("Empty query generated. Skipping.")
            return {
                "instruction": instruction,
                "product_block": "",
                "query": "dummy",
                "index": "1",
                "reward": 0.0,
            } 

        products = self.agent.search(query)
        if not products:
            product_block = ""                
            query    = query
            index =  "1"
            reward        = 0.0
            print("reward",reward)
            return {
                "instruction": instruction,
                "product_block": product_block,
                "query": query,
                "index": index,
                "reward": reward,
            }
        top_10 = products[:10]

        product_lines = []
        for i, p in enumerate(top_10):
            title = p["name"].splitlines()[0]

            price = p.get("price")
            if isinstance(price, (int, float)):
                price_txt = f" – ${price:.2f}"
            else:
                # try to coerce strings like "145.0"
                try:
                    price_txt = f" – ${float(price):.2f}"
                except (TypeError, ValueError):
                    price_txt = ""              # silently drop bad price

            product_lines.append(f"{i+1}. {title}{price_txt}")

        product_block = "\n".join(product_lines)

        prompt_select = [
            {"role": "system", "content": "Pick the best product that matches the instruction."},
            {"role": "user", "content": f"""
        Instruction: {instruction}
        Search Query: {query}

        Here are the top-10 search results:
        {product_block}

        Please respond with **only a single number from 1 to 10**, representing the best matching product. Do not explain or include any other text. Just write the number.
        """}
        ]
        selection = self.model_chat(prompt_select)
        match = re.search(r"\b([1-9]|10)\b", selection)
        if match:
            idx = int(match.group(1)) - 1
        else:
            print("Invalid index, defaulting to 0. Raw output:", selection)
            idx = 0
        print("Index:",idx)
        chosen_product = top_10[idx]
        asin = chosen_product['asin']
        product_details = self.agent.view_product(asin, query, 1, {})
        if product_details:
            options = product_details.get("options", {})
        else:
            options = {}
        purchase = self.agent.buy_product(asin, options)
        reward = purchase['reward'] if purchase else 0.0
        print("Reward:", reward)

    
        return {
            "instruction": instruction,
            "product_block": product_block,
            "query": query,
            "index": idx,
            "reward":    reward,
            }



if __name__ == "__main__":
    model_name = "/home/lokeshk/qwen2.5-1.5b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    SPECIAL = ["<|sep|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL})
    SEP_ID = tokenizer.convert_tokens_to_ids("<|sep|>")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token
    # model.gradient_checkpointing_enable()
    agent = SimpleAgent(base_url="http://10.32.50.50:3000")
    env = WebShopGRPOEnv(agent, tokenizer, model )
    



    training_args = GRPOConfig(
        output_dir="./qwen_fine_tune_model_second",
        run_name="webshop-rl-qwen2.5-1.5b",
        loss_type = "grpo",
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=1,
        num_generations=3,
        num_iterations = 2,
        beta = 0.01,
        max_prompt_length=512,
        max_completion_length=64,
        num_train_epochs=1,
        save_steps=100,
        save_total_limit=5,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        #vllm_gpu_memory_utilization=.3,
        #vllm_device="cuda:0",
        report_to="none", #I'm disabling Wandb.
        optim="paged_adamw_8bit",
        max_steps=20000,
    )

    def final_reward_fn(prompts, completions, **kwargs):
        rewards = kwargs.get("rewards", [0.0] * len(completions))
        return rewards
    
    class DummyDataset(Dataset):
        def __len__(self):          # any positive integer
            return 1

        def __getitem__(self, idx): # value never read by your rollout
            return {"prompt": torch.tensor([0])}
        
    trainer = GRPOEnvTrainer(
        model=model,
        env=env,
        reward_funcs=[final_reward_fn],
        args=training_args,
        processing_class=tokenizer,
        train_dataset=DummyDataset(),
        # optimizers=(AdamW8bit(model.parameters(), lr=5e-6), None)
    )

    trainer.train(resume_from_checkpoint="./qwen_fine_tune_model_second/checkpoint-8300")
    # trainer.train()

    model.save_pretrained("./qwen_fine_tune_model_second")
    tokenizer.save_pretrained("./qwen_fine_tune_model_second")

    print("Fine-tuning completed! Model saved in 'qwen_fine_tune_model_second.")
