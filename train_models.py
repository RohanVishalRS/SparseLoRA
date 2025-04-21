import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from evaluate import load as load_metric
from tqdm import tqdm

from Debertav2Lora import DebertaV2Lora
from RobertaLora import RobertaLora  # Assuming your model is saved in RobertaLora.py
from torch.optim import AdamW


def train_model(dataset, model, epochs=3):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(dataset) * epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(epochs):
        for batch in dataset:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = torch.nn.functional.cross_entropy(logits, batch["label"])

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    return model


def evaluate_model(dataset, model, task):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    metric = load_metric("glue", task)
    total_loss = 0
    total_samples = 0

    model.eval()
    for batch in dataset:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            if task == "stsb":
                loss = torch.nn.MSELoss()(logits.squeeze(), batch["label"].float())
                predictions = logits.squeeze()
            else:
                loss = torch.nn.CrossEntropyLoss()(logits, batch["label"])
                predictions = torch.argmax(logits, dim=-1)

            total_loss += loss.item() * len(batch["label"])
            total_samples += len(batch["label"])

        metric.add_batch(predictions=predictions, references=batch["label"])

    eval_metrics = metric.compute()
    avg_loss = total_loss / total_samples

    if task == "stsb":
        eval_metrics["mse"] = avg_loss

    print(f"\nEvaluation results for {task.upper()}:")
    print(f"Average loss: {avg_loss:.4f}")
    for metric_name, value in eval_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    return eval_metrics, avg_loss

if __name__ == "__main__":
    # Load the GLUE dataset
    task = "sst2"  # Example task: SST-2 (Sentiment Analysis)
    metric = "glue"
    roberta_model_id = "roberta-large"
    deberta_model_id = "microsoft/deberta-v2-xlarge"

    dataset = load_dataset(metric, task)

###################################################################################################################
#   Roberta models
    # Load the tokenizer
    model = RobertaLora(task_name=task, model_id=roberta_model_id)
    encoded_dataset = dataset.map(model.preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenizer = model.tokenizer

    # Prepare data loaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(encoded_dataset["train"], shuffle=True, batch_size=16, collate_fn=data_collator)
    eval_dataloader = DataLoader(encoded_dataset["validation"], batch_size=16, collate_fn=data_collator)

    trained_model = train_model(train_dataloader, model)
    evaluate_model(eval_dataloader, trained_model, task)
###################################################################################################################
###################################################################################################################
#   Deberta models
    model = DebertaV2Lora(task_name=task, model_id=deberta_model_id)

    encoded_dataset = dataset.map(
        model.preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    tokenizer = model.tokenizer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(encoded_dataset["train"], shuffle=True, batch_size=16, collate_fn=data_collator)
    eval_dataloader = DataLoader(encoded_dataset["validation"], batch_size=16, collate_fn=data_collator)

    trained_model = train_model(train_dataloader, model)
    evaluate_model(eval_dataloader, trained_model, task)
###################################################################################################################