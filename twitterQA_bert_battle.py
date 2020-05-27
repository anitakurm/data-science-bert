"""
Script that applies BERT and DistilBERT
to Twitter Q&A dataset.
Tracks answers and running time

To do:
    discuss conceptual choice – loading model for every row
    (to imitate real-world application of Q&A?)
"""
# Import modules
import time
import json
import pandas as pd
import torch
from transformers import BertForQuestionAnswering
from transformers import DistilBertForQuestionAnswering
from transformers import BertTokenizer
from transformers import DistilBertTokenizer


# Import data
filenames = ["TweetQA_data/dev.json", "TweetQA_data/train.json"]

big_df = pd.DataFrame()
for file_name in filenames:
    with open(file_name, "r") as f:
        df = json.load(f)
    df = pd.DataFrame(df)
    # append data from every file to one large dataframe
    big_df = big_df.append(df, ignore_index=True)

# Load both models, track loading time, make a model list
start = time.time()
bert = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
bert_time = time.time() - start

start = time.time()
distilbert = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
distilbert_time = time.time() - start

model_dict = {
    "L_BERT": bert,
    "DistilBERT": distilbert
}

model_loading_times = {
    "L_BERT": str(bert_time),
    "DistilBERT": str(distilbert_time)
}


# Define QA util functions
def bert_QA(row):
    # Get tweet text to fetch answer from
    answer_text = row['Tweet']

    # Get question
    question = row['Question']
    # Start tracking time
    start = time.time()

    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Get start and end scores for answer selection
    start_scores, end_scores = model(torch.tensor([input_ids]))

    # Find the tokens with the highest 'start' and 'end' scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    ans_tokens = input_ids[answer_start:answer_end+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens,
                                                    skip_special_tokens=True)
    # Combine the tokens in the answer and print it out.
    row[answer_id] = tokenizer.convert_tokens_to_string(answer_tokens)
    
    # track time from application of tokenizer to getting the answer
    row[time_id] = str(time.time() - start)
    return row


def distil_bert_QA(row):
    # Get tweet text to fetch answer from
    answer_text = row['Tweet']

    # Get question
    question = row['Question']
    # Start tracking time
    start = time.time()

    # Apply the tokenizer to the input text, treating them as a text-pair.
    encoding = tokenizer.encode_plus(question, answer_text)
    input_ids, att_mask = encoding["input_ids"], encoding["attention_mask"]

    # Get start and end scores for answer selection
    start_scores, end_scores = model(torch.tensor([input_ids]),
                                     attention_mask=torch.tensor([att_mask]))
    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores) + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens,
                                                    skip_special_tokens=True)

    # Combine the tokens in the answer and write it down
    row[answer_id] = tokenizer.convert_tokens_to_string(answer_tokens)
    # track time from application of tokenizer to getting the answer
    row[time_id] = str(time.time() - start)
    return row


for m_name, model in model_dict.items():
    print(f"Running {m_name}")
    answer_id = f"{m_name}_answer"
    time_id = f"{m_name}_time"
    # choose tokenizer
    if m_name == "DistilBERT":
        tok_start = time.time()
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', return_token_type_ids=True)
        distilbert_tok_time = time.time() - tok_start
        big_df = big_df.apply(distil_bert_QA, axis=1)
    else:
        tok_start = time.time()
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        bert_tok_time = time.time() - tok_start
        # Run BERT QA on data, track time
        big_df = big_df.apply(bert_QA, axis=1)

# Write tokenizer loading times down
tokenizer_loading_times = {
    "L_BERT": str(bert_tok_time),
    "DistilBERT": str(distilbert_tok_time)
}

# Write models’ inference and inference time outputs down
big_df.to_csv('twitterQA_berts.csv')

with open('model_loading.txt', 'w') as file:
    file.write(json.dumps(model_loading_times))

with open('tokenizer_loading.txt', 'w') as file:
    file.write(json.dumps(tokenizer_loading_times))
