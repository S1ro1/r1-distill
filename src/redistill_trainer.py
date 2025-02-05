from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from src.config import ScriptConfig
from src.utils import inherit_signature_from
import torch
import torch.amp
from src.dataset import DatasetManager


class RedistillTrainer(Trainer):
    @inherit_signature_from(Trainer.__init__)
    def __init__(
        self,
        teacher_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 512,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.max_seq_length = max_seq_length
        self.preprocessing_class = tokenizer

    def _format_prompt(self, instruction: str) -> str:
        """Format the instruction using the same template"""
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:"
        )

    def get_train_dataloader(self):
        """Custom dataloader that pre-processes the data"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        def preprocess_dataset(examples):
            processed = {}
            prompt = self._format_prompt(examples["instruction"])
            tokenized = self.preprocessing_class(
                prompt,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            processed.update(tokenized)
            return processed

        processed_dataset = self.train_dataset.map(
            preprocess_dataset,
            remove_columns=self.train_dataset.column_names,
            load_from_cache_file=False,
        )
        self.train_dataset = processed_dataset
        return super().get_train_dataloader()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if isinstance(inputs, dict):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
            student_outputs = model(**inputs)

        loss = self._kl_loss(student_outputs, teacher_outputs)

        if return_outputs:
            return (loss, {"loss": loss.item(), "logits": student_outputs.logits})
        return loss

    def _kl_loss(self, student_outputs, teacher_outputs, clamp: int = 100):
        student_logits = student_outputs.logits  # [batch, seq_len, student_vocab]
        teacher_logits = teacher_outputs.logits  # [batch, seq_len, teacher_vocab]

        student_vocab_size = student_logits.size(-1)
        teacher_vocab_size = teacher_logits.size(-1)

        if student_vocab_size < teacher_vocab_size:
            padding = torch.full(
                (
                    student_logits.size(0),
                    student_logits.size(1),
                    teacher_vocab_size - student_vocab_size,
                ),
                float("-inf"),
                device=student_logits.device,
            )
            student_logits = torch.cat([student_logits, padding], dim=-1)

        valid_mask = (student_logits != float("-inf")).float()

        student_logits = torch.clamp(student_logits, -clamp, clamp)
        teacher_logits = torch.clamp(teacher_logits, -clamp, clamp)

        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)

        kl_div = torch.nn.functional.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
            log_target=False,
        )

        masked_kl_div = (kl_div * valid_mask).sum(dim=-1).mean()

        return masked_kl_div


def train_redistill(
    model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    config: ScriptConfig,
) -> AutoModelForCausalLM:
    train_config = config.train_config
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    dataset_manager = DatasetManager(train_config, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    interleaved_dataset = dataset_manager.get_interleaved_dataset()

    if train_config.warmup_ratio is not None:
        warmup_steps = int(dataset_manager.length * train_config.warmup_ratio) // train_config.gradient_accumulation_steps
    else:
        warmup_steps = train_config.warmup_steps // train_config.gradient_accumulation_steps

    training_args = TrainingArguments(
        output_dir=f"outputs/{config.run_name}",
        per_device_train_batch_size=train_config.batch_size,
        learning_rate=train_config.lr,
        warmup_steps=warmup_steps,
        weight_decay=train_config.weight_decay,
        num_train_epochs=train_config.epochs,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        report_to="wandb",
        run_name=config.run_name,
        max_steps=(dataset_manager.length // (train_config.batch_size * train_config.gradient_accumulation_steps))
        * train_config.epochs,
        bf16=True,
        gradient_checkpointing=train_config.gradient_checkpointing,
        gradient_checkpointing_kwargs=train_config.gradient_checkpointing_kwargs,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        optim="adamw_torch_fused",
    )

    trainer = RedistillTrainer(
        model=model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        max_seq_length=train_config.max_seq_length,
        args=training_args,
        data_collator=data_collator,
        train_dataset=interleaved_dataset,
    )

    trainer.train()

    model.push_to_hub(f"siro1/{config.model_name}")
    tokenizer.push_to_hub(f"siro1/{config.model_name}")

    return model
