import os 
import logging 
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch 
import yaml 
import wandb
from peft import LoraConfig, get_peft_model 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from src.helpers.utils import (
    build_train_dataset,
    build_and_load_rag_data,
    tokenizePrompt,
    tokenizePromptAdjustedLengths
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )

def load_config():
    with open('src/helpers/train_config.yaml', 'r') as f:
        return yaml.safe_load(f)
    
def prepare_data(config):
    logging.info("Preparing dataset...")
    train = build_train_dataset(path_1=config['data']['train_path_1'], path_2=config['data']['train_path_2'])
    train_with_rag = build_and_load_rag_data(
        data=train,
        rag_file_path=config['data']['rag_file_path'],
        index_dir=config['data']['index_dir'],
        index_name=config['data']['index_name']
    )

    train_df, eval_df = train_test_split(
        train_with_rag,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=train['Answer_ID']
    )

    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    logging.info(f"Train shape: {train_df.shape}, Eval shape: {eval_df.shape}")

    return Dataset.from_pandas(train_df), Dataset.from_pandas(eval_df)

def prepare_model(config):
    logging.info("Preparing model...")
    bnb_config = BitsAndBytesConfig(**config['bits_and_bytes'])
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(**config['lora'])
    model = get_peft_model(model, lora_config)

    logging.info(f"Trainable parameters: {model.print_trainable_parameters()}")

    return model, tokenizer
def tokenize_data(train_data, eval_data, tokenizer):
    logging.info("Tokenizing data...")
    columns_to_remove = ["question", "answer", "option 1", "option 2", "option 3", "option 4", "option 5", "explanation", "category"]

    tokenize_func = lambda x: tokenizePromptAdjustedLengths(x, tokenizer, max_length=None)
    tokenized_train = train_data.map(tokenize_func, remove_columns=columns_to_remove)
    tokenized_val = eval_data.map(tokenize_func, remove_columns=columns_to_remove)

    return tokenized_train, tokenized_val

def setup_training(config, model, tokenizer, tokenized_train, tokenized_val):
    logging.info("Setting up training...")
    project = config['project']['name']
    model_name = config['model']['name'].replace("\\", "_").replace("/", "_")
    run_name = f"{project}-{model_name}-{config['project']['experiment_version']}"
    output_dir = f"./outputs/{run_name}"

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = TrainingArguments(
        output_dir=output_dir,
        **config['training'],
        logging_dir=f"{output_dir}/logs",
        logging_steps=config['training']['steps_save_eval_loss'],
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=config['training']['steps_save_eval_loss'],
        save_strategy="steps",
        save_steps=config['training']['steps_save_eval_loss'],
        load_best_model_at_end=True,
        report_to="wandb" if config['wandb']['enabled'] else None,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    return trainer

def main():
    setup_logging()
    config = load_config()

    if config['wandb']['enabled']:
        wandb.init(project=config['wandb']['project'], config=config)

    train_data, eval_data = prepare_data(config)
    model, tokenizer = prepare_model(config)
    tokenized_train, tokenized_val = tokenize_data(train_data, eval_data, tokenizer)
    trainer = setup_training(config, model, tokenized_train, tokenized_val)

    logging.info("Starting training...")
    model.config.use_cache = False
    trainer.train()

    logging.info("Saving model...")
    # Create a descriptive model name
    model_name = (
        f"phi-2-teleqa-"
        f"{config['project']['experiment_version']}-"
        f"{config['training']['number_step_partitions'] * config['training']['steps_save_eval_loss']}-steps-"
        f"lr-{config['training']['learning_rate']}"
    )
    
    # Remove any characters that might cause issues in the model name
    model_name = model_name.replace("e-", "e").replace(".", "-")
    
    model.push_to_hub(
        model_name,
        commit_message=f"Training Phi-2 with RAG - {config['project']['experiment_version']}",
        private=True,
        token=os.getenv('HF_TOKEN')
    )
    logging.info(f"Model saved as: {model_name}")
    logging.info("Training completed.")
    if config['wandb']['enabled']:
        wandb.finish()

if __name__ == "__main__":
    main()
