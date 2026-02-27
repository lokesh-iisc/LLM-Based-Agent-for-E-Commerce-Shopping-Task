🛒 LLM-Based Agent for E-Commerce Shopping Tasks
⭐ Research-focused implementation of RL-based LLM agents for grounded decision making.

🚀 M.Tech Thesis Project – IISc Bengaluru
Author: Lokesh Kashyap
Advisor: Prof. Aditya Gopalan


📌 Overview

This project develops a unified LLM-based agent capable of performing goal-driven product search and selection in the WebShop environment (a large-scale e-commerce simulation with 1.18M+ products).
Unlike modular systems, this work trains a single prompt-conditioned policy model to:
🔎 Generate effective search queries
🛍 Select the most relevant product
📈 Improve performance using Reinforcement Learning (GRPO)
The agent learns purely from environment rewards, without any external reward model or verifier.


🧠 Key Contributions

✅ Unified multi-action LLM agent (query + selection using one model)
✅ Prompt-conditioned task switching
✅ Imitation Learning (IL) pretraining
✅ Reinforcement Learning with Group Relative Policy Optimization (GRPO)
✅ Multi-sample, multi-step optimization
✅ End-to-end training with sparse WebShop reward


🏗 Architecture

The system follows this loop:
Instruction → Generate Search Query
Execute Search in WebShop
Retrieve Top-10 Products
Select Best Product
Receive Reward
Update Policy via GRPO
Base Model: Qwen2.5-1.5B-Instruct


📊 Results (WebShop Benchmark)

| Method       | Mean Reward | Score (%) | Success Rate (%) |
| ------------ | ----------- | --------- | ---------------- |
| Rule-Based   | 0.45        | 45.6      | 9.6              |
| Prompt-Based | 0.57        | 57.4      | 26.7             |
| GRPO (Ours)  | **0.63**    | **63.65** | **31.2**         |


⚙ Training Configuration

Generations (G) = 3
Marginals (M) = 3
Iterations (I) = 3
Learning Rate = 2e-5
Optimizer = AdamW
GPU = RTX 2080 Ti


🔬 Technologies Used

Python
PyTorch
Transformers
Supervised Learning
Imitation Learning
Reinforcement Learning
WebShop Environment
GRPO


📈 Future Improvements

Multi-step query refinement
Better attribute grounding
Real-world e-commerce deployment
Improved metadata understanding


👤 Author

Lokesh Kashyap
M.Tech (ECE), IISc Bengaluru
AI | Reinforcement Learning | LLM Agents 
