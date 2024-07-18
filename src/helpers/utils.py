import re
import json
import pandas as pd
import numpy as np
# from open_retrieval.document_loaders import DocumentLoader
# from open_retrieval.text_splitters import TextSplitter
# from open_retrieval.embedding_providers import EmbeddingProvider
# from open_retrieval.vector_databases import VectorDatabase
# from open_retrieval.retrievers import Retriever
# from rerankers import Reranker
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
def build_dataset(parsed_data: json, split: str):
  ids = []
  questions = []
  option_ones = []
  option_twos = []
  option_threes = []
  option_fours = []
  option_fives = []
  answers = []
  explanations = []
  categories = []
  for key, value in parsed_data.items():
    ids.append(key)
    questions.append(value['question'])
    option_ones.append(value['option 1'] if 'option 1' in value else None)
    option_twos.append(value['option 2'] if 'option 2' in value else None)
    option_threes.append(value['option 3'] if 'option 3' in value else None)
    option_fours.append(value['option 4'] if 'option 4' in value else None)
    option_fives.append(value['option 5'] if 'option 5' in value else None)
    answers.append(value['answer'] if split == 'train' else None)
    explanations.append(value['explanation'] if split == 'train' else None)
    categories.append(value['category'] if 'category' in value else None)

  df = pd.DataFrame(columns = ['id', 'question', 'option 1', 'option 2', 'option 3', 'option 4', 'option 5', 'answer', 'explanation', 'category'])
  df['id'] = ids
  df['question'] = questions
  df['option 1'] = option_ones
  df['option 2'] = option_twos
  df['option 3'] = option_threes
  df['option 4'] = option_fours
  df['option 5'] = option_fives
  df['answer'] = answers
  df['explanation'] = explanations
  df['category'] = categories
  return df

def build_train_dataset(path_1: str, path_2: str):
    def load_and_build(path):
        with open(path, 'r') as f:
            data = f.read()
        parsed_data = json.loads(data)
        return build_dataset(parsed_data, 'train')
    
    train_1 = load_and_build(path_1)
    train_2 = pd.read_csv(path_2)[['Question_ID', 'Answer_ID']]

    train_1['Question_ID'] = train_1['id'].str.replace("question", "").astype(int)
    train_2['Answer_ID'] = train_2['Answer_ID'].astype(int)
    train = pd.merge(train_1, train_2, on='Question_ID', how='left').reset_index(drop=True)
    return train

def build_test_dataset(path_1: str, path_2: str):
    def load_and_build(path):
        with open(path, 'r') as f:
            data = json.load(f)
        parsed_data = json.loads(data)
        return build_dataset(parsed_data, 'test')

    test_1 = load_and_build(path_1)
    test_2 = load_and_build(path_2)

    test = pd.concat([test_1, test_2]).reset_index(drop=True)
    return test

def build_and_load_rag_data(data: pd.DataFrame, rag_file_path: str,index_dir: str, index_name: str):
    # if not os.path.exists(rag_file_path):
    #     embedding_provider = 'huggingface'
    #     database = 'faiss'
    #     ranking_model = 'colbert'
    #     ranker = Reranker(ranking_model, verbose = 0)
    #     vector_database = VectorDatabase(vector_store=database)
    #     embedding_function = EmbeddingProvider(embedding_provider=embedding_provider).get_embedding_function( model_name = 'BAAI/bge-base-en-v1.5')
    #     vector_index =  vector_database.create_index(embedding_function=embedding_function, index_name=index_name,index_dir=index_dir)
    #     retriever = Retriever(vector_index, ranker)

    #     for i, row in tqdm(data.iterrows(), total= data.shape[0]):
    #         result = retriever.ranked_retrieval(query= row['question'], top_k = 15, ranked_top_k = 5)
    #         data.loc[i, 'context_1'] = result[0]
    #         data.loc[i, 'context_2'] = result[1]
    #         data.loc[i, 'context_3'] = result[2]
    #         data.loc[i, 'context_4'] = result[3]
    #         data.loc[i, 'context_5'] = result[4]

    #     data.to_csv(rag_file_path, index=False)
    # else:
    data = pd.read_csv(rag_file_path)
    return data

def evaluate(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    unique_labels = np.unique(y_true)
    for label in unique_labels:
        label_accuracy = accuracy_score(y_true[y_true == label], y_pred[y_true == label])
        print(f"Label {label} Accuracy: {label_accuracy:.2f}")

    classification_rep = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(classification_rep)

    confusion_mat = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(confusion_mat)


def formattingFunc(row):
    prompt = "### Context:"
    
    # Adding contexts
    for context in ['context_1', 'context_2', 'context_3']:
        if row.get(context) is not None and row[context].strip():
            prompt += f"{row[context]}\n"
    
    # Adding question
    prompt += f"\n### Question:\n\nBased on the above context: {row['question']}\n\n### Options:\n"

    # Adding options
    for i in range(1, 6):
        option = row.get(f'option {i}')
        if option is not None and option.strip():
            prompt += f"{i}. {option}\n"

    # Adding answer
    prompt += f"\n### Answer: {row['answer']}\n"
    
    return prompt


def tokenizePrompt(prompt:object, tokenizer: object) -> dict:
    tokenizedPrompt = tokenizer(formattingFunc(prompt))
    return tokenizedPrompt
      
      
# this function will set all tokens to the same length using left hand padding and the eos token (setup above)
def tokenizePromptAdjustedLengths(prompt:object, tokenizer:object, maxLengthTokens:int) -> dict:
    """
    Tokenizes prompt with adjusted lengths with left handed padding. All sequences will be of the same length which will assist training.
    """
    tokenizedResponse = tokenizer(
        formattingFunc(prompt),
        truncation=True,
        max_length=maxLengthTokens,
        padding="max_length",
    )
    return tokenizedResponse

def extract_json_from_string(s):
    pattern = re.compile(r'\[.*\]', re.DOTALL)
    match = pattern.search(s)
    if match:
        json_data = match.group()
        return json.loads(json_data)
    return None