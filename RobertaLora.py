from torch import nn
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

from RobertaSelfAttentionLora import RobertaSelfAttentionLora


class RobertaLora(nn.Module):
    def __init__(self, task_name="sst2", dropout_rate=0.1, model_id="roberta-large", lora_rank=8):
        super().__init__()
        self.task_name = task_name
        self.model_id = model_id
        config = RobertaConfig.from_pretrained(model_id, use_pooler=False)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_id)
        self.model = RobertaModel.from_pretrained(model_id, config)
        state_dict = self.model.state_dict()
        self.lora_rank = lora_rank
        self.max_length = self.model.config.max_position_embeddings

        # Task-specific configuration
        self.num_classes = self._get_num_classes(task_name)
        d_model = self.model.config.hidden_size

        # Head for classification/regression
        self.finetune_head_norm = nn.LayerNorm(d_model)
        self.finetune_head_dropout = nn.Dropout(dropout_rate)
        self.finetune_head = nn.Linear(d_model, self.num_classes)

        # Initialize LoRA
        self.replace_attention_with_lora()
        self.model.load_state_dict(state_dict, strict=False)
        self.unfreeze_lora_params()

    def _get_num_classes(self, task_name):
        """Return output dimension based on task."""
        task_to_num_labels = {
            "cola": 2,    # Binary (linguistic acceptability)
            "sst2": 2,    # Binary (sentiment)
            "mrpc": 2,    # Binary (paraphrase)
            "qqp": 2,     # Binary (paraphrase)
            "stsb": 1,    # Regression (similarity score)
            "mnli": 3,    # 3-class (entailment)
            "qnli": 2,    # Binary (question-answer entailment)
            "rte": 2,     # Binary (entailment)
            "wnli": 2,    # Binary (coreference)
            "ax": 3,      # 3-class (MNLI variant)
        }
        return task_to_num_labels.get(task_name, 2)  # Default to binary

    def replace_attention_with_lora(self):
        """Replace attention layers with LoRA variants."""
        for name, module in self.model.named_modules():
            if isinstance(module, RobertaSelfAttention):
                parent = self.model
                *path, layer_name = name.split('.')
                for sub_name in path:
                    parent = getattr(parent, sub_name)
                config = self.model.config
                lora_layer = RobertaSelfAttentionLora(config, rank=self.lora_rank)
                setattr(parent, layer_name, lora_layer)

    def unfreeze_lora_params(self):
        """Only train LoRA parameters (A/B matrices)."""
        for name, param in self.model.named_parameters():
            param.requires_grad = any(k in name for k in ["query_A", "query_B", "value_A", "value_B"])

    def forward(self, input_ids, attention_mask):
        """Forward pass with task-specific output."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.finetune_head_norm(pooled_output)
        pooled_output = self.finetune_head_dropout(pooled_output)
        logits = self.finetune_head(pooled_output)
        return logits

    def preprocess_function(self, examples):
            if self.task_name in ["cola", "sst2"]:
                return self.tokenizer(examples["sentence"], truncation=True, padding="max_length")
            elif self.task_name in ["mrpc", "stsb"]:
                return self.tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")
            elif self.task_name in ["qqp"]:
                return self.tokenizer(examples["question1"], examples["question2"], truncation=True, padding="max_length")
            elif self.task_name in ["mnli", "mnli_mismatched", "mnli_matched"]:
                return self.tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length")
            elif self.task_name in ["qnli"]:
                return self.tokenizer(examples["question"], examples["sentence"], truncation=True, padding="max_length")
            elif self.task_name in ["rte", "wnli"]:
                return self.tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")
            else:
                raise ValueError(f"Unknown task: {self.task_name}")