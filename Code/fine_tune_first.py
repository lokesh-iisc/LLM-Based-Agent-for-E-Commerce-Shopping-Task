import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

MODEL_PATH = "/home/lokeshk/qwen2.5-1.5b-instruct" 
DATASET_PATH = "/home/lokeshk/webshop/baseline_models/data/formatted_search_data.json"  

dataset = load_dataset("json", data_files=DATASET_PATH)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto", use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")

tokenizer.add_special_tokens({"eos_token": "<|endofquery|>"})
model.resize_token_embeddings(len(tokenizer))


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# Tokenization Function
def preprocess_function(examples):
    full_texts = []
    input_lengths = []

    for inst, query in zip(examples["instruction"], examples["query"]):
        # Construct chat-style messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that writes concise e-commerce search queries."},
            {"role": "user", "content": (
                f"Instruction: {inst}\n\n"
                "Now generate only the search query. Do not include any extra explanation or formatting. "
                "Just write the query as a single line."
            )}
        ]
        # Apply chat template and add the query target
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = f"{prompt}{query.strip()} <|endofquery|>"
        full_texts.append(full_text)

        # For loss masking
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_lengths.append(len(prompt_ids))

    # Tokenize full input+target sequences
    tokenized = tokenizer(full_texts, padding="max_length", truncation=True, max_length=128)

    # Create masked labels
    labels = []
    for i, length in enumerate(input_lengths):
        label_ids = tokenized["input_ids"][i].copy()
        label_ids[:length] = [-100] * length  # mask the prompt part
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized




tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./qwen_search_model",
    eval_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    weight_decay=0.001,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=False,
    fp16=False,
    bf16=True,
    max_grad_norm=1.0,
    learning_rate=2e-5,
)

# Train the Model
from transformers import TrainerCallback
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.001)

class NaNCheckCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            grad_norm = logs.get("grad_norm", None)

            print(f"Step {state.global_step}: Loss = {loss:.4f}, Grad Norm = {grad_norm}")

            # If grad_norm is NaN, stop training
            if grad_norm is None or grad_norm != grad_norm:  # NaN check
                print("⚠️ NaN detected in gradients! Stopping training.")
                control.should_training_stop = True

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, None),  # Use custom AdamW optimizer
    callbacks=[NaNCheckCallback()],  #  Check for NaNs and stop training
)


trainer.train()

model.save_pretrained("./qwen_search_model")
tokenizer.save_pretrained("./qwen_search_model")

print("Fine-tuning completed! Model saved in 'qwen_search_model'.")
