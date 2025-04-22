import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from evaluate import load as load_metric
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from Debertav2Lora import DebertaV2Lora
from RobertaLora import RobertaLora
from torch.optim import AdamW

import bitsandbytes as bnb # 8b optim instead of AdamW

def train_model(dataset, model, epochs=3, gradient_accumulation_steps=4, checkpoint_steps=50):
    # optimizer = AdamW(model.parameters(), lr=5e-5)
    optimizer = bnb.optim.Adam(model.parameters(), lr=5e-4, optim_bits=8)
    num_training_steps = (len(dataset) // gradient_accumulation_steps) * epochs
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    scaler = GradScaler()
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    
    save_step = 0  # Track total steps across epochs
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        for step, batch in enumerate(dataset):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast():
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                labels = batch["labels"]
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                progress_bar.update(1)
                
                
                # Save checkpoint every (default 50) steps
                # assuming save_pretrained() works and stuff
                save_step += 1
                '''
                if save_step % checkpoint_steps == 0:
                    checkpoint_path = f"./checkpoints/checkpoint_step_{save_step}"
                    model.save_pretrained(checkpoint_path)
                    model.tokenizer.save_pretrained(checkpoint_path)
                    print(f"\nSaved checkpoint at step {save_step} to {checkpoint_path}")
                '''

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
            with autocast():
                logits = model(input_ids=batch["input_ids"], 
                             attention_mask=batch["attention_mask"])
                labels = batch["labels"]

                if task == "stsb":
                    loss = torch.nn.MSELoss()(logits.squeeze(), labels.float())
                    predictions = logits.squeeze()
                else:
                    loss = torch.nn.CrossEntropyLoss()(logits, labels)
                    predictions = torch.argmax(logits, dim=-1)

                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

        metric.add_batch(predictions=predictions, references=labels)

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
    task = "sst2"
    metric = "glue"
    roberta_model_id = "roberta-large"
    deberta_model_id = "microsoft/deberta-v2-xlarge"

    dataset = load_dataset(metric, task)

###################################################################################################################
#   Roberta models
###################################################################################################################
    print("\nInitializing RoBERTa model...")
    model = RobertaLora(task_name=task, model_id=roberta_model_id)
    
    # modify preprocessing to keep labels because
    # GradScaler refused to work unless I manually set labels like this, but I may be missing something
    # not adding any of the LoSA stuff yet, focused on getting it to run
    # lower epochs w/ gradient accumulation
    def preprocess_with_labels(examples):
        tokenized = model.tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=512)
        tokenized["labels"] = examples["label"]
        return tokenized
    
    encoded_dataset = dataset.map(
        preprocess_with_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer)
    train_dataloader = DataLoader(encoded_dataset["train"], shuffle=True, batch_size=4, collate_fn=data_collator)
    eval_dataloader = DataLoader(encoded_dataset["validation"], batch_size=4, collate_fn=data_collator)
    
    print("\nStarting training...")
    trained_model = train_model(train_dataloader, model, epochs=1, gradient_accumulation_steps=2) 
    evaluate_model(eval_dataloader, trained_model, task)

    # save the model here
    # save_pretrained and save_adapter need to be added later but idk
    '''
    print("\nSaving model...")
    save_path = "./roberta_lora_sst2"
    
    # the entire model (including base weights)
    trained_model.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)
    
    # just the LoRA adapters (smaller files):
    trained_model.save_adapter(save_path, "lora_sst2")
    
    print(f"Model saved to {save_path}")
    '''
    
###################################################################################################################
#   Deberta models
#   removed until later cus i just wanted the first one to run
#   but the 1st one takes to long to run anyways
###################################################################################################################
    '''
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
    '''