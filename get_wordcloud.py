###############################################################################
# This script writes wordclouds                                               #
###############################################################################

# last major updates November 2024
# contact fredstroessi@gmail.com

### Load packages -----------------------------------------------
import pandas as pd
import os # paths
import ftfy # fix encodings
import matplotlib.pyplot as plt # plot

from gensim.utils import simple_preprocess
import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load("de_core_news_sm")

# Randomize the document order
from random import shuffle

# Next, create the term dictionary
from gensim import corpora

# Train LDA with 10 topics and print
from gensim.models.ldamodel import LdaModel

# plot
from numpy.random import randint
from wordcloud import WordCloud


### Set directory  --------------------------------------------------
new_directory = "/home/onyxia/work"
os.chdir(new_directory)
os.getcwd()



### Load data and aggregate pages to full text and sample --------------------------------------------------

# read df and transform text entries to string
df = pd.read_csv('./Pages_49.csv',sep=';', encoding='utf-8')
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

# Set a seed (random_state) for replicability
df_sample = aggregated_df
print(len(df_sample))

# Fix the encodings
df_sample['alltexts'] = [ftfy.fix_text(i) for i in list(df_sample['alltexts'])]



### Preprocessing  --------------------------------------------------

def tokenize(x, nlp):
    # lemmatize and lowercase without stopwords, punctuation and numbers
    return [w.lemma_.lower() for w in nlp(x) if not w.is_stop and not w.is_punct and not w.is_digit and len(w) > 2]

# Clean the texts
# Attention: takes a couple of minutes!
text_clean = [tokenize(i, nlp) for i in list(df_sample['alltexts'])]

# Randomize the document order
from random import shuffle
shuffle(text_clean)



### Dictionary, filter and LDA training  --------------------------------------------------

# Next, create the term dictionary
from gensim import corpora
dictionary = corpora.Dictionary(text_clean)

# Filter the extremes: drop all words that appear in less than 10 paragraphs in every tweet or more
dictionary.filter_extremes(no_below=10, no_above=0.99, keep_n=1000)
print (len(dictionary))

# Then, create the document term matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_clean]

# Train LDA with 10 topics and print
from gensim.models.ldamodel import LdaModel
lda = LdaModel(doc_term_matrix, num_topics=10, id2word = dictionary, passes=3)
lda.show_topics(formatted=False)


output_file = "lda_topics.txt"

# Write the string to a text file
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(text)


### Plot the wordcloud  --------------------------------------------------

from numpy.random import randint
from wordcloud import WordCloud

plt.rcParams['figure.figsize'] = [20, 15]
# make word clouds for the topics

for i,weights in lda.show_topics(num_topics=-1, num_words=100, formatted=False):
    maincol = randint(0,360)
    def colorfunc(word=None, font_size=None, 
                  position=None, orientation=None, 
                  font_path=None, random_state=None):   
        color = randint(maincol-10, maincol+10)
        if color < 0:
            color = 360 + color
        return "hsl(%d, %d%%, %d%%)" % (color,randint(65, 75)+font_size / 7, randint(35, 45)-font_size / 10)   

    wordcloud = WordCloud(background_color="white", 
                          ranks_only=False, 
                          max_font_size=120,
                          color_func=colorfunc,
                          height=600,width=800).generate_from_frequencies(dict(weights))

    plt.subplot(3, 4, i+1)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # Save the plot as an image file
    plt.savefig('cloud49.png', bbox_inches='tight', dpi=150)
