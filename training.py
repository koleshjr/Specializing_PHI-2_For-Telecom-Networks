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

    train_df, eval_df = train_test_split(
        train_with_rag,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=train['Answer_ID']
    )

    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    logging.info("Train shape: {}, Eval shape: {}".format(train_df.shape, eval_df.shape))

    return Dataset.from_pandas(train_df), Dataset.from_pandas(eval_df)

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
def tokenize_data(train_data, eval_data, tokenizer):
    logging.info("Tokenizing data...")
    columns_to_remove = ["question", "answer", "option 1", "option 2", "option 3", "option 4", "option 5", "explanation", "category"]

    # Format and Tokenize datasets.
    tokenizedTrain = train_data.map(lambda x: tokenizePrompt(x, tokenizer))
    tokenizedVal = eval_data.map(lambda x: tokenizePrompt(x, tokenizer))

    # count lengths of both datasets so we can adjust max length
    lengthTokens:list=[len(x['input_ids']) for x in tokenizedTrain] # count lengths of tokenizedTrain
    if tokenizedVal != None:
        lengthTokens += [len(x['input_ids']) for x in tokenizedVal] # count lengths of tokenizedVal
    maxLengthTokens:int=max(lengthTokens) + 2 #  we could also visualise lengthTokens using matplotlib if we wish to see the distribution
    tokenDiffOriginal:int=maxLengthTokens-min(lengthTokens) # create metric original
    logging.info("maxLengthTokens: {},tokenDiffOriginal: {}".format(maxLengthTokens, tokenDiffOriginal))

    del tokenizedTrain; del tokenizedVal # clean up old variables

    tokenize_func = lambda x: tokenizePromptAdjustedLengths(x, tokenizer, maxLengthTokens=maxLengthTokens)
    tokenized_train = train_data.map(tokenize_func, remove_columns=columns_to_remove)
    tokenized_val = eval_data.map(tokenize_func, remove_columns=columns_to_remove)

    return tokenized_train, tokenized_val

def setup_training(config, model, tokenizer, tokenized_train, tokenized_val):
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
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=config['training']['steps_save_eval_loss'],
        save_strategy="steps",
        save_steps=config['training']['steps_save_eval_loss'],
        load_best_model_at_end=True,
        report_to="wandb" if config['wandb']['enabled'] else None,
        warmup_steps=config['training']['warmup_steps'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        optim=config['training']['optim'],
        num_train_epochs= config['training']['num_train_epochs']
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
    # random_seed(2022,True)
    setup_logging()
    config = load_config()

    if config['wandb']['enabled']:
        wandb.init(project=config['wandb']['project'], config=config)

    train_data, eval_data = prepare_data(config)
    model, tokenizer = prepare_model(config)
    tokenized_train, tokenized_val = tokenize_data(train_data, eval_data, tokenizer)
    trainer = setup_training(config, model, tokenizer, tokenized_train, tokenized_val)

    logging.info("Starting training...")
    model.config.use_cache = False
    trainer.train()

    logging.info("Saving model...")
    # Create a descriptive model name
    model_name = (
    "phi-2-teleqa-"
    "{}-"
    "{}-epochs-"
    "lr-{}".format(
        config['project']['experiment_version'],
        config['training']['num_train_epochs'],
        config['training']['learning_rate'],
    )
    )

    # Remove any characters that might cause issues in the model name
    model_name = model_name.replace("e-", "e").replace(".", "-")
    
    model.push_to_hub(
        model_name,
        commit_message="Training Phi-2 with RAG - {}".format(config['project']['experiment_version']),
        private=True,
        token=os.getenv('HF_TOKEN')
    )
    logging.info("Model saved as: {}".format(model_name))
    logging.info("Training completed.")
    if config['wandb']['enabled']:
        wandb.finish()

if __name__ == "__main__":
    main()
