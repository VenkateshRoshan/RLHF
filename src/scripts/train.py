import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Load Dataset
raw = load_dataset("Anthropic/hh-rlhf")
chosen = raw["train"]["chosen"]
rejected = raw["train"]["rejected"]
print(f"Loaded hh-rlhf dataset: chosen={len(chosen)}, rejected={len(rejected)}")
# For speed/demo purposes only, take a small subset if dataset is large
chosen = chosen[:256]
rejected = rejected[:256]


# Convert to a simple (text, label) dataset for reward model training
texts = []
labels = []
for t in chosen:
    texts.append(t)
    labels.append(1.0)
for t in rejected:
    texts.append(t)
    labels.append(0.0)

dataset = Dataset.from_dict({"text": texts, "labels": labels})

reward_model_name = "bert-base-uncased"
reward_tokenizer = BertTokenizer.from_pretrained(reward_model_name)
reward_model = BertForSequenceClassification.from_pretrained(reward_model_name, num_labels=1)

def tokenize_reward(examples):
    out = reward_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    out["labels"] = [float(l) for l in examples["labels"]]
    return out


dataset = dataset.map(tokenize_reward, batched=True)

os.makedirs("./reward_model", exist_ok=True)
args = TrainingArguments(output_dir="./reward_model", per_device_train_batch_size=4, num_train_epochs=1, logging_steps=10)
trainer = Trainer(model=reward_model, args=args, train_dataset=dataset)
trainer.train()

# Trainer will write checkpoints under ./reward_model; find the most recent checkpoint
def get_latest_checkpoint(path):
    if not os.path.isdir(path):
        return path
    # look for checkpoint-* folders
    entries = [os.path.join(path, d) for d in os.listdir(path) if d.startswith("checkpoint-")]
    if not entries:
        return path
    entries.sort(key=os.path.getmtime, reverse=True)
    return entries[0]

latest = get_latest_checkpoint("./reward_model")
print(f"Loading reward model from: {latest}")
reward_model = BertForSequenceClassification.from_pretrained(latest)
reward_tokenizer = BertTokenizer.from_pretrained(reward_model_name)


# --- Setup policy model (GPT-2) for generation / PPO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# gpt2 tokenizer has no pad token by default; set it to eos and use left padding for causal models
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.padding_side = 'left'

# ensure reward model is on the same device as inputs/policy
reward_model = reward_model.to(device)


def score_responses(responses):
    # responses: list[str]
    inputs = reward_tokenizer(responses, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = reward_model(**inputs)
        # outputs.logits shape: (batch, 1) for regression/regression-like label
        logits = outputs.logits.squeeze(-1)
        # Return Python floats
        return logits.cpu().tolist()


def generate_and_score(prompts, max_new_tokens=64):
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    out = policy_model.generate(**batch, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    responses = tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    rewards = score_responses(responses)
    return responses, rewards


if __name__ == "__main__":
    prompts = ["Explain reinforcement learning in simple terms:", "How do transformers work?"]

    responses, rewards = generate_and_score(prompts)
    for p, r, rew in zip(prompts, responses, rewards):
        print("PROMPT:", p)
        print("RESPONSE:", r)
        print("REWARD:", rew)
        print("---")

    # Optional: run a short PPO loop if TRL and correct configuration are available
    try:
        ppo_config = PPOConfig(
            model_name_or_path="gpt2",
            batch_size=2,
            forward_batch_size=1,
            learning_rate=1.41e-5,
            log_with=None,
        )
        print("Initializing PPOTrainer (this may require TRL and proper environment)...")
        ppo_trainer = PPOTrainer(ppo_config, policy_model, tokenizer=tokenizer)

        # Small demo step: generate responses and compute rewards, then call step
        # Note: the exact API for PPOTrainer.step may vary by TRL version; adapt as needed.
        queries = prompts
        responses, rewards = generate_and_score(queries)
        # Convert to lists and call ppo step if available
        try:
            # ppo_trainer expects lists of queries and responses and rewards per sample
            loss = ppo_trainer.step(queries, responses, rewards)
            print("PPO step finished; loss:", loss)
        except Exception as e:
            print("Could not run ppo_trainer.step() — TRL API mismatch or environment issue:", e)
    except Exception as e:
        print("Skipping PPOTrainer setup — TRL may not be available or config failed:", e)
