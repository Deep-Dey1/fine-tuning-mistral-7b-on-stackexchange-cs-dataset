import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import datasets
import wandb
import evaluate
import numpy as np

# Initialize WandB
wandb.init(project="llm_finetuning", name="Mistral7B_Finetuning")

# Load model and tokenizer (Mistral 7B or any selected model)
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,  # Using QLoRA for efficiency
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply LoRA Configuration
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load dataset
DATASET_NAME = "stackexchange/cs"  # Specify your dataset
train_data = datasets.load_dataset(DATASET_NAME, split="train")
val_data = datasets.load_dataset(DATASET_NAME, split="validation")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    report_to="wandb",
    save_total_limit=2,
    load_best_model_at_end=True,  # For early stopping
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10
)

# Define Trainer
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

# Evaluate the model
predictions, labels, _ = trainer.predict(val_data)
predictions = np.argmax(predictions, axis=-1)

precision = evaluate.load("precision").compute(predictions=predictions, references=labels)
recall = evaluate.load("recall").compute(predictions=predictions, references=labels)
f1 = evaluate.load("f1").compute(predictions=predictions, references=labels)

wandb.log({"precision": precision, "recall": recall, "f1_score": f1})

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(labels, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

print("Training and evaluation complete!")
