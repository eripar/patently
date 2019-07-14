
# Import Flask and pandas
from flask import Flask, render_template, request
import pandas as pd
import datetime

# Import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Import spaCy for NLP
import spacy
import string
from spacy.lang.en import English
from nltk.stem.porter import *
stemmer = PorterStemmer()

# Load the spaCy english library, punctuations, and stopwords
nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation
parser = English()
# Define the stop-words to filter
stop_words = spacy.lang.en.stop_words.STOP_WORDS
# add a shortlist of patent stop-words
nlp.Defaults.stop_words |= {"electrochemical","electrochemistry","use","comprise", "invention","composition",\
                            "form","include","active","method","compound","group","high","present","provide",\
                            "produce","contain"}

# Read the patent dataset
df_user_search= pd.read_csv("data/user_search.csv",)
df = pd.read_csv("data/export_patent_view_main_tokenized_v2.csv",)
df_descriptions= pd.read_csv("data/class_descriptions.csv",)
copy_df = df

# Define the TFIDF vectorizer
vectorizer=TfidfVectorizer(ngram_range=(1,3))
# Imported dataset contains the patent abstracts pre-processed (using spacy_tokenizer function below) into tokens as a column.  Use that column as the features.
docs = copy_df['tokens']
bag = vectorizer.fit_transform(docs)
length = len(docs)

app = Flask(__name__)

# Function to process the user's input into tokens that can be compared against the tokenized patent abstract data
def spacy_tokenizer(docs):  
    mytokens = []
    mytokens_out = []
    print_tokens = []
    mytokens = parser( docs )
    # Process text NO1: lowercase, remove spaces, and lemmatize
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ !='-PRON-' else word.lower_ for word in mytokens ]
    # Process text NO2: remove stop words and punctuations
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ] 
    
    mytokens = nlp( str( mytokens )) 
    for word in mytokens: 
        #remove verbs, punctuation, conjunctions, adverbs, numbers, and undefined parts-of-speech
        mytoken = (word.pos_!='VERB') & (word.pos_!='PUNCT') & (word.pos_!='CCONJ') & (word.pos_!='ADV') & (word.pos_!='NUM') & (word.pos_!='X')
        if mytoken:
            print_tokens.append(word)
            #stem
            #word=stemmer.stem(word.text)
            mytokens_out.append(word)
            
    mytokens_out = pd.Series(str(mytokens_out))
    return mytokens_out, print_tokens


# initial page front
@app.route("/")
def home():
    return render_template("index.html.j2")


# page that returns after a user "submits"
@app.route("/analyze", methods=['POST'] )
def analyze():
    copy_df = df
    if request.method == "POST":
        # Bring in the text query and number of results wanted by the user
        rawtext = request.form["rawtext"]
        n = request.form["result_count"]
        n = int(n)
        received_text = rawtext
        
        #store user search for analysis
        df_user_search= pd.read_csv("data/user_search.csv",)
        df_store = pd.DataFrame( {'timestamp':[datetime.datetime.now()], 'user_input':[rawtext] } )
        df_user_search = pd.concat([df_user_search, df_store], ignore_index=True, sort=True)
        df_user_search = df_user_search[['timestamp', 'user_input']]
        df_user_search.to_csv("data/user_search.csv",)
        
        #convert input into tokens for the user to see
        text_tokens, print_tokens = spacy_tokenizer(received_text)
        #vectorizer user input
        user_input = pd.Series(text_tokens)
        #transform tokenized query into TFIDF
        test = vectorizer.transform(user_input)
        #test cosine similarities against all the documents
        cosine_similarities = linear_kernel(test, bag).flatten()
        #return indices of most similar documents
        related_docs_indices = cosine_similarities.argsort()[:-(n+1):-1]
        #store similarity in the dataframe
        copy_df['similarity'] = cosine_similarities
        
        #extract the top n to be returned to the index.html
        out_df = copy_df.iloc[list(related_docs_indices)][['title', 'abstract','patent_id', 'similarity', 'year','subclass_id','url']]
        #reset the indeces for the returned data
        out_df = out_df.reset_index()
        #convert dataframe into dictionary
        out_df = out_df.to_dict('records')
        
    return render_template('analyze.html.j2', received_text=rawtext, text_tokens=print_tokens[:], table=out_df )

#https://www.websiteout.net/counter.php


if __name__ == "__main__":
    app.run(debug=True, port=5957)
