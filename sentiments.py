###############################################################################
# This script imports newspapers, cleans and assigns sentiments               #
###############################################################################

# last major updates November 2024
# contact fredstroessi@gmail.com

### Load packages -----------------------------------------------

import pandas as pd
import os # paths
import spacy
from spacy_sentiws import spaCySentiWS # incorporate Leipzig corpora in Spacy to do sentiment analysis
from string import punctuation
import nltk
from nltk.tokenize import word_tokenize




### Set directory  --------------------------------------------------
new_directory = "/home/onyxia/work"
os.chdir(new_directory)
os.getcwd()



# read df and transform text entries to string
df = pd.read_csv('./Pages_48.csv',sep=';', encoding='utf-8')
df['plainpagefulltext'] = df['plainpagefulltext'].apply(lambda x: str(x) if x is not None else "")

# Ensure the date column is in datetime format, then extract year_month identifier and get city names
df["publication_date"] = pd.to_datetime(df["publication_date"])  
df["year_month"] = df["publication_date"].dt.to_period("M") 
df["city"] = df["place_of_distribution"]

# aggregate all pages of a paper
aggregated_df = df.groupby(["paper_title", "zdb_id", "city", "year_month", "publication_date"]).agg(
    alltexts=("plainpagefulltext", " ".join)  # Concatenate texts for each city and month
).reset_index()

# get a sample
aggregated_df  = aggregated_df.sample(frac=0.1, random_state=123).reset_index(drop=True)

# call cleaner function and apply
cleaner = HistoricalGermanOCRCleaner()
aggregated_df['alltexts_clean'] = aggregated_df['alltexts'].apply(cleaner.clean_text)
print(aggregated_df)
aggregated_df.to_csv("./Test.csv",sep=';', encoding='utf-8') 
aggregated_df = pd.read_csv('./Test.csv',sep=';', encoding='utf-8')


print(aggregated_df)


### Further pre-processing  --------------------------------------------------
# remove punctuation
#nltk.download('punkt_tab')
#punc_remover = str.maketrans('','',punctuation) 
#aggregated_df['alltexts'] = [i.translate(punc_remover) for i in list(aggregated_df.alltexts)]
#tokenize
#aggregated_df['alltexts_tokenized'] = [word_tokenize(i) for i in list(aggregated_df.alltexts)]
#aggregated_df.to_csv("C:/Users/freds/Documents/02Studium/00 MA/Data/Newpaper sentiments/AggrTest.csv",sep=';', encoding='utf-8') 
#print(aggregated_df)
#aggregated_df['alltexts_tokenized'][391]




### Sentiments  --------------------------------------------------

# relevant packages
import re
import spacy
import pandas as pd
import numpy as np
from spacy_sentiws import spaCySentiWS 

# Load spaCy German model with SentiWS
nlp = spacy.load('de_dep_news_trf') #or de_core_news_sm
#sentiws = spaCySentiWS(nlp)
nlp.add_pipe('sentiws', config={'sentiws_path': '.'})

## cut windows around target words and perform sentiment analysis within the windows
# Define keywords and window size globally
KEYWORDS = ['Demokratie', 'Freiheit','Wahl']
WINDOW_SIZE = 50



aggregated_df['windows']=aggregated_df['alltexts'].apply(extract_windows_from_text)
aggregated_df['windows'][5]

aggregated_df['sentiment_analysis'] = aggregated_df['windows'].apply(analyze_text_windows)
print(aggregated_df)

# declare lists
mean_sentiments = []
median_sentiments = []
mean_scores=[]
median_scores=[]

for sentiment_results in aggregated_df['sentiment_analysis']:
    # window means and medians aggregated
    mean_sentiment_values = [window['mean_sentiment'] for window in sentiment_results if 'mean_sentiment' in window]
    median_sentiment_values = [window['median_sentiment'] for window in sentiment_results if 'median_sentiment' in window]
    mean_sentiments.append(np.mean(mean_sentiment_values) if mean_sentiment_values else np.nan)
    median_sentiments.append(np.mean(median_sentiment_values) if median_sentiment_values else np.nan)
    # all scores and taking mean and median aggregating windows
    all_scores = [score for window in sentiment_results if 'sentiment_scores' in window for score in window['sentiment_scores']]
    # Calculate the mean and median of these scores
    if all_scores:
        mean_scores.append(np.mean(all_scores))
        median_scores.append(np.median(all_scores))
    else:
        mean_scores.append(np.nan)
        median_scores.append(np.nan)
aggregated_df['window_mean_mean'] = mean_sentiments
aggregated_df['window_median_mean'] = median_sentiments
aggregated_df['all_mean'] = mean_scores

# Save results to a new CSV
aggregated_df.to_csv('newspaper_sentiment_sample.csv', index=False,sep=';', encoding='utf-8')
aggregated_df = pd.read_csv('./newspaper_sentiment_sample.csv',sep=';', encoding='utf-8')

aggregated_df['windows'][9]


from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("mdraw/german-news-sentiment-bert")
model = AutoModelForSequenceClassification.from_pretrained("mdraw/german-news-sentiment-bert")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
aggregated_df['windows'][17]
classifier(aggregated_df['windows'][162])

for idx, windows in enumerate(aggregated_df['windows']):
    for window in windows:
        text = window['text']
        result = classifier(text)[0]
        window['label'] = result['label']
        window['score'] = result['score']

X_train = ["Hallo, ich finde das richtig gut.Wieso denn nicht",
"Was geht ab", "das ist schon eher schlecht", "Freiheit wurde massiv untergraben","Düstere stimmung ist schlecht"]

classifier(X_train)
print(df)


res = classifier(X_train)
print(res)



# create a text file
df = aggregated_df
all_texts = " ".join(df['alltexts'].tolist())  # Joining all text entries into one string
all_texts.to_text('texts.csv', index=False,sep=';', encoding='utf-8')
output_file = "output_text.txt"

# Write the string to a text file
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(all_texts)

# Print some example results
print(df[['plainpagefulltext', 'aggregated_window_text', 'mean_sentiment', 'median_sentiment']].head())





check = correct_text_with_ocronos(aggregated_df['windows'][17],tokenizer,model)

# Read the OCR text from the file
with open("./output_text.txt", "r", encoding="utf-8") as file:
    text_content = file.read()






# call and apply cleaner function from other script, then save cleaned df
cleaner = HistoricalGermanOCRCleaner()
df['plainpagefulltext_clean'] = df['plainpagefulltext'].apply(cleaner.clean_text)
df.to_csv("C:/Users/freds/Documents/02Studium/00 MA/Data/Newpaper sentiments/Test.csv",sep=';', encoding='utf-8') 

# load previously saved dataframe and transform text entries to strings
df = pd.read_csv('C:/Users/freds/Documents/02Studium/00 MA/Data/Newpaper sentiments/Test.csv',sep=';', encoding='utf-8')
df['plainpagefulltext_clean'] = df['plainpagefulltext_clean'].apply(lambda x: str(x) if x is not None else "")
df['plainpagefulltext'] = df['plainpagefulltext'].apply(lambda x: str(x) if x is not None else "")
df['plainpagefulltext_clean'][2971]

df['plainpagefulltext'][1]




### Collapse texts by city and year-month  --------------------------------------------------

aggregated_df = df.groupby(["city", "year_month"]).agg(
    alltexts=("plainpagefulltext", " ".join)  # Concatenate texts for each city and month
).reset_index()
print(aggregated_df)
