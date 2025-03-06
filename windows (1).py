###############################################################################
# This script imports newspapers, cleans and assigns sentiments               #
###############################################################################

# last major updates December 2024
# contact fredstroessi@gmail.com

### Load packages -----------------------------------------------

import pandas as pd
import os # paths
put "C:/Users/freds/Documents/02Studium/00 MA/Data/Newspaper raw/Pages_63.csv" "/home/903690/Pages_63.csv"  


import re
from typing import List, Dict, Union

### Set directory  --------------------------------------------------
new_directory = "/home/onyxia/work"
os.chdir(new_directory)
os.getcwd()


# read df and transform text entries to string
df = pd.read_csv('./Pages_63.csv',sep=';', encoding='utf-8')
df['plainpagefulltext'] = df['plainpagefulltext'].apply(lambda x: str(x).replace('ſ', 's').replace('=', '-').replace('< space >', '   ')  if x is not None else "")
df['plainpagefulltext'][1]
# Ensure the date column is in datetime format, then extract year_month identifier and get city names
df["publication_date"] = pd.to_datetime(df["publication_date"])  
df["year_month"] = df["publication_date"].dt.to_period("M") 
df["year"] = df["publication_date"].dt.year
df["city"] = df["place_of_distribution"]

conversion = pd.read_csv('./Preuss_ZDB.csv',sep=',', encoding='utf-8').drop(['Unnamed: 0'],axis=1).drop_duplicates()#.drop_duplicates(subset=['zdb_id'])
print(conversion)

# aggregate all pages of a paper
aggregated_df = df.groupby(["year","paper_title", "zdb_id", "city", "year_month", "publication_date"]).agg(
    alltexts=("plainpagefulltext", " ".join)  # Concatenate texts for each city and month
).reset_index().merge(conversion, on=['zdb_id','zdb_id'], how='inner')
print(aggregated_df)




## Split text into chunks of 100 words

def split_into_chunks(text, chunk_size=100):
    words = text.split()  # Split text into words
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0,len(words),chunk_size)]
    return chunks

# Apply the function to split texts in the DataFrame
aggregated_df['subtexts'] = aggregated_df['alltexts'].apply(lambda x: split_into_chunks(x))

# Explode the chunks into separate rows (optional, if you need one chunk per row)
aggregated_df = aggregated_df.explode('subtexts', ignore_index=True)
print(aggregated_df)

# target_revolution_words =[
#     #"verfassung", #7000
#     #"constitu"
#     #"freiheit",
#     #"amerika",
#     #"revolution" ##oft
#     #"demokrat"
#     #"Liberty"
#     #"slave",
#     #"sklave"
#     #"kirche"
#     "könig",
# ]
#exclusion_patterns = []

#patriotismus

target_words = [
    # "re gierung",
    # "regierung"
    # "opposition",
    # "patriotismus",
    # "minorität",  # with possible variation: mino'ität
    # "mino'ität",
    # "majorität",
    # "bundestag",
    # "ministerium",
    # "minister",

    #"Proletariat"

    # "revolutionair",  # corrected spelling
    # "corruption",   
    # "haushalts etats",     

#"majestät",
    # "se . majestät",
    # "se . maj",
  #   "ma jestät"
   # "prinz",
    # "erlaucht",
   #  "könig ",
   #  "koenig ",
    
    # "hoheit",
   # "aristokrat",
    #"adel",
    #"führer",
    #"herzog",
    #"fürst"
    #"Lärm rechts;"
    # "König Friedrich",
    # "König von Preu",
    # #"Friedrich Wilhelm"
    # "Preußischen Hofes",
    # "Preußischer Hofes",
    # "Friedrich Wilhelm IV",
    # "hohenzolle"
    #
    "Kirche"
    #"Verfassung"
]

# List of exclusion patterns (e.g., "spanische Regierung")
exclusion_patterns = [
#     r'\bspanische regierung\b', r'\bengl. regierung\b', r'\bsardinische regierung\b', r'\bengli sche regierung\b',r'\bprinz napoleon\b',
#     r'\bengl. regierung\b',r'\bengl. regierung\b',r'\b britische. regierung\b',r'\b britische regierung\b', 
#     r'\bholländischen Regierung\b',r'\bholländische Regierung\b',r'\bRegierung der vereinigten Staaten\b',
#     r'\bfranzösische Regierung\b',r'\bKönig von Nea pel\b', r'\bKönig von Neapel\b',r'\böſterreichiſche Regierung\b',r'\böſterreichische Regierung\b',r'\bTory = Regierung',r'\brussischen Regierung\b',
#     r'\bnapoletanische Regierung\b',r'\bnapoletanischen Regierung\b',r'\bNapoleons Regierung\b',r'\bKönig von Sardinien\b'
  ]




# # Function to filter texts
# def filter_texts(text, target_words, exclusion_patterns):
#     target_word_matches = [word for word in target_words if re.search(fr'\b{word}\b', text, re.IGNORECASE)]
#     if target_word_matches:
#         excluded = any(re.search(pattern, text, re.IGNORECASE) for pattern in exclusion_patterns)
#         if not excluded or len(target_word_matches) > 1:
#             return True
#     return False

# # Apply the filtering function
# aggregated_df['contains_target'] = aggregated_df['subtexts'].apply(
#     lambda x: filter_texts(x, target_words, exclusion_patterns)
# )




def filter_texts(
    text: str,
    word_combinations: List[List[str]],
    exclusion_patterns: List[str]
) -> bool:
    # First check exclusions
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in exclusion_patterns):
        return False
    # Check for word combinations
    for word_group in word_combinations:
        # Check if all words in this combination are present
        if all(re.search(fr'\b{word}\b', text, re.IGNORECASE) for word in word_group):
            return True
    return False

# Example usage:
word_combinations = [
    #["König", "Preu"],  # Both these words must appear
    #["Koenig", "Preu"]  # OR both these words must appear
    #"Reibungen",
    #["unruhe"],
    #["franz", "revolution"],
    #["franz", "re volution"]
    #["Napoleon"]
    # ["protest"]#,reform revolution
    #["1848","könig"]
    #["könig","preu"],
    #["könig", "friedrich"],
    #["regent"]
    #["absolutismus"],
    #["Le gitimation"],
    #["Legitimation"]
    #["Wilhelm I"],
    ["liberal"],#konservative,  linksliberale Deutsche Fortschrittspartei, National - Versammlung
    ["Libe ral"],
    ["konstitution"],
    ["verfassung"]

]

# Apply the filtering function
aggregated_df['contains_target'] = aggregated_df['subtexts'].apply(
    lambda x: filter_texts(x, word_combinations, exclusion_patterns)
)



# Filter rows where the condition is met
filtered_df = aggregated_df[aggregated_df['contains_target']].reset_index(drop=True).drop('contains_target', axis=1)
print(filtered_df)


filtered_df.drop('alltexts', axis=1).to_csv("./Test.csv",sep=';', encoding='utf-8') 
filtered_df.drop('alltexts', axis=1).sample(frac=0.01, random_state=123).reset_index(drop=True).to_csv("./arist.csv",sep=';', encoding='utf-8') 






# Define keywords and window size globally for extract_contact_windows()
KEYWORDS = ['demokratie', 'freiheit','frei heit','wahl','volk', 'verfassung', 'Pressefreiheit', 
'volksversammlung', 'volks versammlung', 'Paulskirche', 'Paulskirchen',
'deutsch', 'deutschland',  'versammlung']
