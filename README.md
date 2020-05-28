# DistilBERT vs BERT on TweetQA data ⚗️
Code for our exam project for the Data Science course as a part of MSc in Cognitive Science at Aarhus University. 

In this paper the large version of BERT (Bidirectional Encoder Representations from Transformers by [Huggingface 🤗](https://github.com/huggingface/transformers)) is compared to its smaller knowledge-distilled version DistilBERT. Both models were fine-tuned to the SQuAD 1.1 task. The aim was to see how well these pre-made models perform on a new dataset that hasn’t been part of their training nor fine-tuning data - the [TweetQA dataset](https://tweetqa.github.io/) by Xiong et al. (2019).

## Getting Started

### Prerequisites & Installing

Make a virtual environment and install necessary Python modules in your Terminal:

```
virtualenv bert --python /usr/bin/python3
source bert/bin/activate
pip install -r requirements.txt
```
In your R seesion, install the package manager pacman using command: 
``` 
install.packages("pacman")
```


## Description
* The script [twitterQA_bert_battle.py](twitterQA_bert_battle.py) applies both models to train and dev parts of the TweetQA dataset. 
 The script outputs the [model inferences and processing times](processed_data/twitterQA_berts.csv), [model loading times](processed_data/model_loading.txt) and [tokenizer loading times ](processed_data/tokenizer_loading.txt)

* The notebook [Evaluation.ipynb](Evaluation.ipynb) applies automated evaluation metrics GLEU and METEOR scores to model inferences, and outputs [GLEU scores for questions with short answers](processed_data/df_short_answers.csv) and [METEOR scores for questions with longer answers](processed_data/df_long_answers.csv)  

* The R Markdown [visualisations.Rmd](visualisations.Rmd) uses files in [processed_data](processed_data) to visualise results and run simple t-tests to compare two models


## Built With
Python, Jupyter Notebook and R



## Versioning
See [requirements.txt](requirements.txt)

## Authors
Anita Kurm and Maris Sala 


## Acknowledgments
* [Huggingface 🤗](https://github.com/huggingface/transformers)
* [TweetQA dataset](https://tweetqa.github.io/)
