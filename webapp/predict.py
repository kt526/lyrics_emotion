# import tensorflow as tf
from unicodedata import normalize
from bs4 import BeautifulSoup
import contractions
from nltk import *
from unicodedata2 import *
from inflect import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle
warnings.filterwarnings('ignore')

def text_preprocessing_platform(test_lyrics, remove_stopwords=True):
    ## Define functions for individual steps
    # First function is used to denoise text
    def denoise_text(text):
        # Strip html if any. For ex. removing <html>, <p> tags
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        # Replace contractions in the text. For ex. didn't -> did not
        text = contractions.fix(text)
        return text
    
    def remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words
    
    def to_lowercase(words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words
    
    def remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words
    
    def replace_numbers(words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words
    
    def remove_stopwords(words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words
    
    def stem_words(words):
        """Stem words in list of tokenized words"""
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems
    
    def lemmatize_verbs(words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas
    
    ### A wrap-up function for normalization
    def normalize_text(words, remove_stopwords):
        words = remove_non_ascii(words)
        words = to_lowercase(words)
        words = remove_punctuation(words)
        words = replace_numbers(words)
        if remove_stopwords:
            words = remove_stopwords(words)
        #words = stem_words(words)
        words = lemmatize_verbs(words)
        return words

    # Tokenize tweet into words
    def tokenize(text):
        return word_tokenize(text)

    # A overall wrap-up function
    def text_prepare(text):
        text = denoise_text(text)
        text = ' '.join([x for x in normalize_text(tokenize(text), remove_stopwords)])
        return text
    
    testing = text_prepare(test_lyrics)
    
    # return processed df
    return testing


def predict_emotion(raw_lyrics):
    processed_lyrics = [text_preprocessing_platform(raw_lyrics, remove_stopwords=False)]

    # load the model from disk
    cv = pickle.load(open('models/fitted_cv.pkl', 'rb'))
    X_lyrics_bow = cv.transform(processed_lyrics).toarray()
    # print(X_lyrics_bow)

    model = pickle.load(open('models/finalized_model.pkl', 'rb'))
    model_y_pred = model.predict(X_lyrics_bow)
    emotion = get_emotion(model_y_pred)

    return emotion

def get_emotion(emotion):
    # Define the emotion_map dictionary
    emotion_map = {
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprise'
    }

    # Map the value to the corresponding emotion using the emotion_map dictionary
    mapped_emotion = emotion_map[emotion[0]]
    return mapped_emotion


