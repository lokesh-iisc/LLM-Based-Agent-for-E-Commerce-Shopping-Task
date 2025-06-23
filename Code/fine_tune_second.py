import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

MODEL_PATH = "/home/klokesh/qwen2.5-1.5b-instruct"
DATA_PATH = "/home/klokesh/webshopp/baseline_models/data/your_detailed_dataset.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load tokenizer and dataset ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
raw_data = load_dataset("json", data_files=DATA_PATH)['train']

# === Preprocessing for imitation ===
class ChoiceImitationDataset(Dataset):
    def __init__(self, data):
        self.samples = []
        for ex in data:
            query = ex["query"]
            products = ex["products"]
            label_indices = [i for i, p in enumerate(products) if p["label"] == 1]
            if not label_indices:
                continue
            label = label_indices[0]
            candidates = [
                f"Title: {p['title']} | Price: {p['price']} | Attrs: {', '.join(p['attributes'])}"
                for p in products
            ]
            self.samples.append((query, candidates, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# === Collator for batching ===
def collate_fn(batch):
    queries, candidate_lists, labels = zip(*batch)
    flat_candidates = [cand for candidates in candidate_lists for cand in candidates]
    flat_query_repeats = [q for q, candidates in zip(queries, candidate_lists) for _ in candidates]
    
    query_inputs = tokenizer(flat_query_repeats, return_tensors="pt", padding=True, truncation=True, max_length=512)
    cand_inputs = tokenizer(flat_candidates, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    labels = torch.tensor(labels)
    candidate_counts = [len(c) for c in candidate_lists]
    
    return query_inputs, cand_inputs, labels, candidate_counts

# === Model ===
class QwenChoiceModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.cross_attn = nn.MultiheadAttention(self.encoder.config.hidden_size, num_heads=8, batch_first=True)
        self.W = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, query_inputs, cand_inputs, candidate_counts):
        with torch.no_grad():
            query_outputs = self.encoder(**query_inputs).last_hidden_state  # (B*N, T, H)
            cand_outputs = self.encoder(**cand_inputs).last_hidden_state    # (B*N, T, H)

        attn_output, _ = self.cross_attn(cand_outputs, query_outputs, query_outputs)  # Cross-attn
        pooled = attn_output.mean(dim=1)  # Mean pooling
        scores = self.W(pooled).squeeze(-1)  # Scalar score per candidate

        # Group scores per original batch item
        batched_scores = []
        idx = 0
        for count in candidate_counts:
            batched_scores.append(scores[idx:idx + count])
            idx += count
        return batched_scores

# === Loss function ===
def compute_loss(batched_scores, labels):
    losses = []
    for score, label in zip(batched_scores, labels):
        log_probs = F.log_softmax(score, dim=0)
        losses.append(-log_probs[label])
    return torch.stack(losses).mean()

# === Training loop (basic) ===
def train():
    dataset = ChoiceImitationDataset(raw_data)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = QwenChoiceModel(MODEL_PATH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        model.train()
        total_loss = 0
        for step, (query_inputs, cand_inputs, labels, counts) in enumerate(loader):
            for k in query_inputs:
                query_inputs[k] = query_inputs[k].to(device)
                cand_inputs[k] = cand_inputs[k].to(device)

            labels = labels.to(device)
            batched_scores = model(query_inputs, cand_inputs, counts)
            loss = compute_loss(batched_scores, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        print(f">>> Epoch {epoch} Avg Loss: {total_loss / len(loader):.4f}")

train()
