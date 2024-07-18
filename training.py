import os 
import random
import numpy as np
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
from dotenv import load_dotenv
from src.helpers.utils import (
    build_train_dataset,
    build_and_load_rag_data,
    tokenizePrompt,
    tokenizePromptAdjustedLengths
)
load_dotenv()
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )

def random_seed(seed_value, use_cuda): 
    np.random.seed(seed_value)
 #cpu vars
    torch.manual_seed(seed_value) 
# cpu  vars
    random.seed(seed_value)
 # Python 
    if use_cuda: 
        torch.cuda.manual_seed(seed_value) 
        torch.cuda.manual_seed_all(seed_value) 
# gpu vars
        torch.backends.cudnn.deterministic = True 
 #needed
        torch.backends.cudnn.benchmark = False 
#Remember to use num_workers=0 when creating the DataBunch.


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

    logging.info("Dataset shape: {}".format(train_with_rag.shape))

    return Dataset.from_pandas(train_with_rag)


def prepare_model(config):
    logging.info("Preparing model...")
    print(config['bits_and_bytes'])
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

    logging.info("Trainable parameters: {}".format(model.print_trainable_parameters()))

    return model, tokenizer
def tokenize_data(data, tokenizer):
    logging.info("Tokenizing data...")
    columns_to_remove = ["question", "answer", "option 1", "option 2", "option 3", "option 4", "option 5", "explanation", "category"]

    # Format and Tokenize dataset
    tokenized_data = data.map(lambda x: tokenizePrompt(x, tokenizer))

    # count lengths of dataset so we can adjust max length
    lengthTokens = [len(x['input_ids']) for x in tokenized_data]
    maxLengthTokens = max(lengthTokens) + 2
    tokenDiffOriginal = maxLengthTokens - min(lengthTokens)
    logging.info("maxLengthTokens: {}, tokenDiffOriginal: {}".format(maxLengthTokens, tokenDiffOriginal))

    del tokenized_data  # clean up old variable

    tokenize_func = lambda x: tokenizePromptAdjustedLengths(x, tokenizer, maxLengthTokens=maxLengthTokens)
    tokenized_data = data.map(tokenize_func, remove_columns=columns_to_remove)

    return tokenized_data

def setup_training(config, model, tokenizer, tokenized_data):
    logging.info("Setting up training...")
    project = config['project']['name']
    model_name = config['model']['name'].replace("\\", "_").replace("/", "_")
    run_name = "{}-{}-{}".format(project, model_name, config['project']['experiment_version'])

    output_dir = "./outputs/{}".format(run_name)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir="{}/logs".format(output_dir),
        logging_steps=config['training']['steps_save_eval_loss'],
        do_eval=False,  # No evaluation
        save_strategy="steps",
        save_steps=config['training']['steps_save_eval_loss'],
        report_to="wandb" if config['wandb']['enabled'] else None,
        warmup_steps=config['training']['warmup_steps'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        optim=config['training']['optim'],
        num_train_epochs=config['training']['num_train_epochs']
    )
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_data,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    return trainer

def main():
    setup_logging()
    config = load_config()

    if config['wandb']['enabled']:
        wandb.init(project=config['wandb']['project'], config=config)

    data = prepare_data(config)
    model, tokenizer = prepare_model(config)
    tokenized_data = tokenize_data(data, tokenizer)
    trainer = setup_training(config, model, tokenizer, tokenized_data)

    logging.info("Starting training...")
    model.config.use_cache = False
    trainer.train()

    logging.info("Saving model...")
    # Create a descriptive model name
    model_name = (
    "phi-2-teleqa-"
    "{}-"
    "{}-epochs-"
    "lr-{}-"
    "full-dataset".format(
        config['project']['experiment_version'],
        config['training']['num_train_epochs'],
        config['training']['learning_rate'],
    )
    )

    # Remove any characters that might cause issues in the model name
    model_name = model_name.replace("e-", "e").replace(".", "-")
    
    model.push_to_hub(
        model_name,
        commit_message="Training Phi-2 with RAG - {} - Full Dataset".format(config['project']['experiment_version']),
        private=True,
        token=os.getenv('HF_TOKEN')
    )
    logging.info("Model saved as: {}".format(model_name))
    logging.info("Training completed.")
    if config['wandb']['enabled']:
        wandb.finish()

if __name__ == "__main__":
    main()
