# What time is it? Temporal Analysis of Novels

This repo contains the code and a link to the dataset for EMNLP 2020 submission: What time is it? Temporal Analysis of Novels.

## Data
We only provide data for the [Gutenberg time dataset](https://drive.google.com/file/d/1bTE2Ul9maAji5B1YVnCYqWUOytXxxq-5/view?usp=sharing) and not for HathiTrust due to copyright laws.

Below is a description of the Gutenberg time dataset:
- `guten_id` - Gutenberg ID number
- `hour_reference` - hour from 0 to 23
- `time_phrase` - the phrase corresponding to the referenced hour
- `is_ambiguous` - boolean whether it is clear whether time is AM or PM
- `time_pos_start` - token position where `time_phrase` begins
- `time_pos_end` - token position where `time_phrase` ends (exclusive)
- `tok_context` - context in which `time_phrase` appears as space-separated tokens

## Requirements
Depending on which models you want to run, not all of these libraries are required. `sklearn` is used for linear models and train/test splitting. The LSTM model uses `gensim` and `keras` and the BERT model uses `torch` and `transformers`.

- `sklearn==0.23.2`
- `gensim==3.8.0`
- `keras=2.3.1`
- `torch==1.6.0`
- `transformers==3.2.0`

## How to Run
To run our models, edit the `config.ini` file with the appropriate paths. You will need to download the [Gutenberg time dataset](https://drive.google.com/file/d/1bTE2Ul9maAji5B1YVnCYqWUOytXxxq-5/view?usp=sharing) as well as the txt file for [GloVe embedding](http://nlp.stanford.edu/data/glove.6B.zip) for the LSTM model.

- `time_csv` : path to where you stored the Gutenberg time dataset
- `glove_input_file` : path to where you stored the [GloVe embedding txt file](http://nlp.stanford.edu/data/glove.6B.zip) (we use `glove.6B.300d.txt`)
- `time_imputed_csv` : path to where the imputed dataset will be stored (created when running `disambiguate_am_pm.py` in the `am_pm_prediction` directory)



