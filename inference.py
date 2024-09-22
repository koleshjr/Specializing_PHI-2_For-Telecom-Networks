import os
import csv
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
import shutil
from src.helpers.utils import testformattingFunc, get_answer_id
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig)
from peft import PeftModel

def main():
    load_dotenv()
    original_test = pd.read_csv('data/test_with_rag_ranked_filtered_2k_512.csv')
    model_name = "microsoft/phi-2"
    finetunedFolder = "Koleshjr/phi-2-teleqa-1-0-0-5-epochs-lr-0-001-full-dataset-512-1k"
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type='nf4',
                                    bnb_4bit_compute_dtype='float16',
                                    bnb_4bit_use_double_quant=True)
    
    modelInference = AutoModelForCausalLM.from_pretrained(model_name,
                                                           torch_dtype=torch.float32,
                                                           device_map="auto",
                                                           trust_remote_code=True,
                                                           quantization_config=bnb_config)

    tokenizerInference = AutoTokenizer.from_pretrained(model_name,
                                                       add_bos_token=True,
                                                       trust_remote_code=True,
                                                       use_fast=False)
    tokenizerInference.pad_token = tokenizerInference.eos_token

    FTmodel = PeftModel.from_pretrained(modelInference, finetunedFolder)

    original_test['prompt'] = original_test.apply(testformattingFunc, axis=1)

    if os.path.exists('data/test_with_rag_ranked_filtered_progress_1k_5epochs_2k_512_model.csv'):
        shutil.copy('data/test_with_rag_ranked_filtered_progress_1k_5epochs_2k_512_model.csv',
                    'data/test_with_rag_ranked_filtered_progress_backup_1k_5epochs_2k_512_model.csv')
        mod_test = pd.read_csv('data/test_with_rag_ranked_filtered_progress_1k_5epochs_2k_512_model.csv', on_bad_lines='skip', engine='python')
    else:
        mod_test = original_test.copy()
        mod_test['response'] = ""

    # Identify skipped rows
    skipped_rows = original_test[~original_test['id'].isin(mod_test['id'])]

    # Concatenate skipped rows to mod_test
    test = pd.concat([mod_test, skipped_rows], ignore_index=True)

    null_responses = test[test['response'].isnull() | (test['response'] == "")]
    repetition_penalty = 1.0
    max_tokens = 50

    for i, row in tqdm(null_responses.iterrows(), total=null_responses.shape[0]):
        FTmodel.eval()
        with torch.no_grad():
            tokenisedPrompt = tokenizerInference(row['prompt'], return_tensors="pt").to("cuda")
            response = tokenizerInference.decode(FTmodel.generate(**tokenisedPrompt, max_new_tokens=max_tokens, repetition_penalty=repetition_penalty)[0], skip_special_tokens=True)
            test.loc[i, 'response'] = response
            test.to_csv('data/test_with_rag_ranked_filtered_progress_1k_5epochs_2k_512_model.csv', index=False)

    test['Task'] = "Phi-2"
    test['Answer_ID'] = test['response'].apply(get_answer_id)
    test['Question_ID'] = test['id'].str.replace('question', '').astype(int)
    test['Answer_ID'] = np.where(test['Answer_ID'] == 32, 3, test['Answer_ID'])
    test['Answer_ID'] = test['Answer_ID'].fillna(4)
    test['Answer_ID'] = test['Answer_ID'].astype(int)
    submission = test[['Question_ID', 'Answer_ID', 'Task']]
    submission.to_csv('data/submission_with_rag_ranked_filtered_progress_1k_5epochs_2k_512_model.csv', index=False)

if __name__ == "__main__":
    main()
