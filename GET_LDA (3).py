import pandas as pd
import os # paths
import pickle
import re
from typing import List, Dict, Union
from random import shuffle

import spacy
from spacy.tokenizer import Tokenizer

from numpy.random import randint
from wordcloud import WordCloud
import matplotlib.pyplot as plt # plot

from gensim.utils import simple_preprocess
from gensim import corpora # dictionary
from gensim.models.ldamodel import LdaModel


### Set directory  --------------------------------------------------
new_directory = "/home/onyxia/work"
os.chdir(new_directory)
os.getcwd()

### Set up spacy pipeline  --------------------------------------------------

#nlp = spacy.load("de_core_news_sm", disable=["parser", "ner", "tagger"])


### Load data  --------------------------------------------------

# df40_46 = pd.read_pickle('./Preprocessed40_46.pkl')
# df47_50 = pd.read_pickle('./Preprocessed47_50.pkl')
# df51_54 = pd.read_pickle('./Preprocessed51_54.pkl')
# df55_58 = pd.read_pickle('./Preprocessed55_58.pkl')
# df59_62 = pd.read_pickle('./Preprocessed59_62.pkl')
# df63_67 = pd.read_pickle('./Preprocessed63_67.pkl')
# df_sample = pd.concat([df40_46,df47_50, df51_54,df55_58,df59_62,df63_67])
#df_sample = df_sample#.sample(frac=0.0001, random_state=123).reset_index(drop=True)

## Keep only rows where clean_tokens is a non-empty list
# df_sample = df_sample[df_sample['clean_tokens'].map(len) > 0]

# df_sample.to_pickle("./Preprocessed.pkl")



### Ensure numbers and undesired chars are removed  -----------------------------

# Filter tokens to remove any word containing numbers
# with open("Preprocessed.pkl", "rb") as f:
#     df_sample = pickle.load(f)

# import string
# from string import punctuation

## Create punctuation remover translation table
# punc_remover = str.maketrans('', '', punctuation)

## clean tokens
# df_sample['clean_tokens'] = df_sample['clean_tokens'].apply(
#     lambda tokens: [
#         cleaned_word for word in tokens
#         if isinstance(word, str)
#         and word.lower() != "ver" and word.lower() != "sch" and word.lower() != "ser"
#         and word.lower() != "ter" and word.lower() != "—" and word.lower() != "ger" and word.lower() != "lich"
#         and word.lower() != "ten" and word.lower() != "½"
#         and len(cleaned_word := ''.join([char for char in word.translate(punc_remover) if not char.isdigit()])) > 1
#     ] if isinstance(tokens, list) else []
# )

## remove empty tokens:
# df_sample['clean_tokens'] = df_sample['clean_tokens'].apply(
#     lambda tokens: [word for word in tokens if word] if isinstance(tokens, list) else []
# )

## save
# df_sample.to_pickle("./Preprocessed_final.pkl")

## Load prepared data
with open("Preprocessed_final.pkl", "rb") as f:
    df_sample = pickle.load(f)

df_sample = df_sample.reset_index(drop=True)
df_sample['paper_id'] = df_sample.index  
print(df_sample)

### Split data to 100 token chunks  --------------------------------------------------

def split_token_list(tokens, chunk_size=100):
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
df_sample['token_chunks'] = df_sample['clean_tokens'].map(lambda x: split_token_list(x, chunk_size=100))
df_chunks = df_sample.explode('token_chunks', ignore_index=True)
df_chunks = df_chunks.drop('clean_tokens', axis=1)
# Add chunk number within each paper
df_chunks['chunk_nr'] = df_chunks.groupby('paper_id').cumcount()
#df_chunks['paper_chunk_id'] = df_chunks.apply(lambda row: f"{row['paper_id']}_{row['chunk_number']}", axis=1)
print(df_chunks['token_chunks'][1])



del df_sample
import gc
gc.collect()
#df_chunks.drop(columns=['token_chunks','clean_tokens']).to_csv("./CHUNKS_NO_INFO.csv", sep=';', encoding='utf-8', index=False)



### Prepare LDA training  --------------------------------------------------#

# text_clean = df_chunks['token_chunks'].tolist()
# with open("text_clean.pkl", "wb") as f:
#     pickle.dump(text_clean, f)
# with open("text_clean.pkl", "rb") as f:
#     text_clean = pickle.load(f)


## Randomize the document order
# import random
# from random import shuffle
# random.seed(123) 
# shuffle(text_clean)
# print(text_clean[:60]) 



### Dictionary, Document term matrix --------------------------------------------------

## Term dictionary
from gensim import corpora
# dictionary = corpora.Dictionary(text_clean)
# dictionary.save('dictionary40_67.dict')
dictionary = corpora.Dictionary.load('dictionary40_67.dict')

## Filter extremes: drop all words that appear in less than 10 paragraphs in every tweet or more
dictionary.filter_extremes(no_below=10, no_above=0.99, keep_n=1000)
print (len(dictionary))

## Then, create the document term matrix
# doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_clean]
# with open("corpus40_67.pkl", "wb") as f:
#     pickle.dump(doc_term_matrix, f)
# with open("corpus40_67.pkl", "rb") as f:
#     doc_term_matrix = pickle.load(f)
# print(doc_term_matrix[:5]) 


### LDA training --------------------------------------------------

## Train LDA with 10 topics and print
from gensim.models.ldamodel import LdaModel
# lda = LdaModel(doc_term_matrix, num_topics=16, id2word = dictionary, passes=3)
# lda.show_topics(formatted=False)
# lda.save("lda_model_40_67_T16_NONUM.gensim")
lda = LdaModel.load("./ModelT16/lda_model_40_67_T16.gensim")

## Print top ten keywords per topic
# output_file = "lda_topics16.txt"
# with open(output_file, 'w', encoding='utf-8') as file:
#     for topic in lda.print_topics():
#         file.write(f"{topic}\n")



## Coherence plots
# from gensim.models.coherencemodel import CoherenceModel
# import gensim
# import matplotlib.pyplot as plt
# coherencemodel = CoherenceModel(model=lda, texts=text_clean, dictionary=dictionary, coherence='c_v')
# coherencemodel.get_coherence()

## Define the function to compute coherence values
# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=5, random_state=42)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     return model_list, coherence_values

## Run the coherence calculation (adjust parameters as needed)
# start, limit, step = 2, 40, 3
# model_list, coherence_values = compute_coherence_values(
#     dictionary=dictionary,
#     corpus=doc_term_matrix,
#     texts=text_clean,
#     start=start,
#     limit=limit,
#     step=step
# )

## Plot the coherence scores
# x = range(start, limit, step)
# plt.figure(figsize=(10,6))
# plt.plot(x, coherence_values, marker='o')
# plt.xlabel("Number of Topics")
# plt.ylabel("Coherence Score")
# plt.title("Optimal Number of Topics (LDA)")
# plt.xticks(x)
# plt.grid(True)
# plt.show()






### Plot Wordcloud  --------------------------------------------------

# from numpy.random import randint
# from wordcloud import WordCloud
# lda = LdaModel.load("./ModelT16/lda_model_40_67_T16_NONUM.gensim")

# ## set size
# plt.rcParams['figure.figsize'] = [20, 15] ## for 12 max topics
# #plt.rcParams['figure.figsize'] = [32, 24] ## for 30 topics

# ## plot wordcloud
# for i,weights in lda.show_topics(num_topics=-1, num_words=100, formatted=False):
#     maincol = randint(0,360)
#     def colorfunc(word=None, font_size=None, 
#                   position=None, orientation=None, 
#                   font_path=None, random_state=None):   
#         color = randint(maincol-10, maincol+10)
#         if color < 0:
#             color = 360 + color
#         return "hsl(%d, %d%%, %d%%)" % (color,randint(65, 75)+font_size / 7, randint(35, 45)-font_size / 10)   
#     wordcloud = WordCloud(background_color="white", 
#                           ranks_only=False, 
#                           max_font_size=120,
#                           color_func=colorfunc,
#                           height=600,width=800).generate_from_frequencies(dict(weights))
#     #plt.subplot(3, 4, i+1) ## for 12 max topics
#     #plt.subplot(4, 5, i+1) ## for 12 max topics
#     plt.subplot(4, 4, i+1) ## for 30 topics
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
#     # Save the plot as an image file
#     plt.savefig('cloud40_67_T16.png', bbox_inches='tight', dpi=150)
#     #plt.savefig(f'cloud56_topic_{i}.png', bbox_inches='tight', dpi=150)



### Visualize the topics  --------------------------------------------------#

# lda = LdaModel.load("./lda_model_40_67_T12_NONUM.gensim")

# import pyLDAvis.gensim_models as gensimvis
# import pyLDAvis
# pyLDAvis.enable_notebook()
# vis = gensimvis.prepare(lda, doc_term_matrix, dictionary)
# vis

# # Save as standalone HTML file
# pyLDAvis.save_html(vis, 'lda_visualization_40_67_T12.html')



### Load data from previous steps -------------------------------------------

## Load Preprocessed data
# with open("Preprocessed_final.pkl", "rb") as f:
#     df_sample = pickle.load(f)
# print(df_sample)

## Load Trained LDA model
# from gensim.models.ldamodel import LdaModel
# lda = LdaModel.load("./ModelT12/lda_model_40_67_T12.gensim")

## create an unshuffled version of the text-lists
# with open("text_clean.pkl", "rb") as f:
#     text_clean = pickle.load(f)

## create an unshuffled version of the Document term matrix and save it
# doc_term_matrix_unshuffled = [dictionary.doc2bow(doc) for doc in text_clean]
# with open("corpus40_67_UNSHUFFLED.pkl", "wb") as f:
#     pickle.dump(doc_term_matrix_unshuffled, f)

## Load the unshuffled document term matrix
with open("corpus40_67_UNSHUFFLED.pkl", "rb") as f:
    doc_term_matrix_unshuffled = pickle.load(f)

## Load dictionary
from gensim import corpora
dictionary = corpora.Dictionary.load('dictionary40_67.dict')
dictionary.filter_extremes(no_below=10, no_above=0.99, keep_n=1000) # Filter the extremes: drop all words that appear in less than 10 paragraphs in every tweet or more





# ### Topic to DATA -------------------------------------------

# def format_topics_sentences(ldamodel, corpus, texts_df):
#     topics_data = []
#     # Efficiently iterate over corpus, ldamodel output, and dataframe rows simultaneously
#     for doc_topics, (_, row) in zip(ldamodel[corpus], texts_df.iterrows()):
#         #for doc_topics, text in zip(ldamodel[corpus], texts):
#         dominant_topic, perc_contribution = max(doc_topics, key=lambda x: x[1])
#         topic_keywords = ", ".join([word for word, _ in ldamodel.show_topic(dominant_topic)])
#         topics_data.append([
#             int(dominant_topic),
#             round(perc_contribution, 4),
#             topic_keywords,
#             row['token_chunks'],  # use token_chunks here
#             row['year']           # include year here
#         ])
#     df_topics = pd.DataFrame(
#         topics_data,
#         columns=['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Year']
#     )
#     return df_topics


# ## Test on a subsample using first 100 rows
# df_chunk_sample = df_chunks.head(10).reset_index(drop=True)
# doc_term_matrix_sample = doc_term_matrix_unshuffled[:10]

# ## Apply function
# df_topic_sents_keywords = format_topics_sentences(
#     ldamodel=lda, 
#     corpus=doc_term_matrix_sample, 
#     texts_df=df_chunk_sample
# )

# ## Format and save
# df_dominant_topic = df_topic_sents_keywords.reset_index()
# df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Year']

# # Save aligned topics
# df_dominant_topic.to_csv("./topics12.csv", sep=';', encoding='utf-8', index=False)# head(150).to_csv("./topics12.csv", sep=';', encoding='utf-8', index=False)


df_chunk_sample = df_chunks.head(10).reset_index(drop=True)
doc_term_matrix_sample = doc_term_matrix_unshuffled[:10]

# def format_topics_sentences_full(ldamodel, corpus, texts_df):
#     topics_data = []
#     num_topics = ldamodel.num_topics

#     for doc_idx, (doc_topics, (_, row)) in enumerate(zip(ldamodel[corpus], texts_df.iterrows())):
#         # Sort topics by weight
#         doc_topics_sorted = sorted(doc_topics, key=lambda x: x[1], reverse=True)

#         # Dominant topic and its keywords
#         dominant_topic, dominant_contribution = doc_topics_sorted[0]
#         topic_keywords = ", ".join([word for word, _ in ldamodel.show_topic(dominant_topic)])

#         # Create a dictionary of topic contribution for all topics
#         topic_contributions = {f"Topic_{i}": 0.0 for i in range(num_topics)}
#         for topic_num, weight in doc_topics:
#             topic_contributions[f"Topic_{topic_num}"] = round(weight, 4)

#         # Assemble row
#         topics_data.append({
#             'document_no': doc_idx,
#             'zdb_id': row.get('zdb_id'),
#             'paper_id':row.get('paper_id'),
#             'chunk_nr':row.get('chunk_nr'),
#             'year': row.get('year'),
#             'city': row.get('city'),
#             'year_month': row.get('year_month'),
#             'publication_date': row.get('publication_date'),
#             'paper_title': row.get('paper_title'),
#             'Dominant_Topic': dominant_topic,
#             'Dominant_Topic_Contribution': round(dominant_contribution, 4),
#             'Keywords': topic_keywords,
#             **topic_contributions  # merge in topic columns
#         })

#     # Final dataframe
#     df_full = pd.DataFrame(topics_data)
#     return df_full

# df_topic_full = format_topics_sentences_full(
#     ldamodel=lda,
#     corpus=doc_term_matrix_sample,
#     texts_df=df_chunk_sample
# )

# # Save to CSV
# df_topic_full.to_csv("./topics_full.csv", sep=';', encoding='utf-8', index=False)






def format_topics_sentences_fast(ldamodel, corpus, texts_df):
    from tqdm import tqdm
    tqdm.pandas()

    num_topics = ldamodel.num_topics
    all_doc_topics = list(ldamodel.get_document_topics(corpus, minimum_probability=0.0))
    rows = texts_df.to_dict('records')

    topics_data = []

    for doc_idx, (doc_topics, row) in enumerate(zip(all_doc_topics, rows)):
        doc_topics_sorted = sorted(doc_topics, key=lambda x: x[1], reverse=True)
        dominant_topic, dominant_contribution = doc_topics_sorted[0]
        topic_keywords = ", ".join([word for word, _ in ldamodel.show_topic(dominant_topic)])

        topic_contributions = {f"Topic_{i}": 0.0 for i in range(num_topics)}
        for topic_num, weight in doc_topics:
            topic_contributions[f"Topic_{topic_num}"] = round(weight, 4)

        topics_data.append({
            'document_no': doc_idx,
            'zdb_id': row.get('zdb_id'),
            'paper_id': row.get('paper_id'),
            'chunk_nr': row.get('chunk_nr'),
            'year': row.get('year'),
            'city': row.get('city'),
            'year_month': row.get('year_month'),
            'publication_date': row.get('publication_date'),
            'paper_title': row.get('paper_title'),
            'Dominant_Topic': dominant_topic,
            'Dominant_Topic_Contribution': round(dominant_contribution, 4),
            'Keywords': topic_keywords,
            **topic_contributions
        })

    return pd.DataFrame(topics_data)

df_topic_full = format_topics_sentences_fast(
    ldamodel=lda,
    corpus=doc_term_matrix_unshuffled,
    texts_df=df_chunks
)

# Save to CSV
df_topic_full.to_csv("./topics_full.csv", sep=';', encoding='utf-8', index=False)


# Add helper column: is Topic 4 dominant
df_topic_full['is_topic4_dominant'] = df_topic_full['Dominant_Topic'] == 4

# Group by zdb_id and year
summary = (
    df_topic_full
    .groupby(['zdb_id', 'year','city'])
    .agg(
        topic4_dominant_share=('is_topic4_dominant', 'mean'),      # Share of dominant topic 4
        topic4_mean_contribution=('Topic_4', 'mean'),              # Average contribution of topic 4
        num_chunks=('paper_id', 'count'),                          # Total chunks
        num_papers=('paper_id', pd.Series.nunique)                 # Unique papers
    )
    .reset_index()
)

summary.to_csv("./topic_4.csv", sep=';', encoding='utf-8', index=False)
