import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from afinn import Afinn
from flair.models import TextClassifier
from flair.data import Sentence
import torch
import emoji

nltk.download('vader_lexicon')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sia = SentimentIntensityAnalyzer()
transformer_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
afinn = Afinn()
flair_sentiment = TextClassifier.load('sentiment-fast')
emotion_pipeline = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")
emotion_text_cnn_pipeline = pipeline("sentiment-analysis", model="bhadresh-savani/distilbert-base-uncased-emotion")

tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

deepmoji_model = AutoModelForSequenceClassification.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
deepmoji_tokenizer = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

def get_vader_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def get_textblob_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def get_transformer_sentiment(text):
    result = transformer_pipeline(text)[0]
    if result['label'] == '5 stars':
        return 1.0
    elif result['label'] == '4 stars':
        return 0.75
    elif result['label'] == '3 stars':
        return 0.5
    elif result['label'] == '2 stars':
        return -0.5
    elif result['label'] == '1 star':
        return -1.0
    else:
        return 0.0

def get_afinn_sentiment(text):
    return afinn.score(text) / 10.0

def get_flair_sentiment(text):
    if not text.strip():
        return 0.0
    sentence = Sentence(text)
    flair_sentiment.predict(sentence)
    if not sentence.labels:
        return 0.0
    sentiment = sentence.labels[0]
    if sentiment.value == 'POSITIVE':
        return sentiment.score
    else:
        return -sentiment.score

def get_combined_sentiment(text):
    vader_score = get_vader_sentiment(text)
    textblob_score = get_textblob_sentiment(text)
    transformer_score = get_transformer_sentiment(text)
    afinn_score = get_afinn_sentiment(text)
    flair_score = get_flair_sentiment(text)

    combined_score = (vader_score * 0.4 + textblob_score * 0.3 + transformer_score * 0.2 + afinn_score * 0.05 + flair_score * 0.05)
    return combined_score

def get_emotion_sentiment(text):
    result = emotion_pipeline(text)[0]
    return result

def get_emotion_text_cnn_sentiment(text):
    result = emotion_text_cnn_pipeline(text)[0]
    return result

def get_goemotions_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 
              'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 
              'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 
              'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    max_score_idx = torch.argmax(scores).item()
    return labels[max_score_idx], scores[0][max_score_idx].item()

def extract_emojis(text):
    return [char for char in text if char in emoji.EMOJI_DATA]

def get_deepmoji_sentiment(text):
    inputs = deepmoji_tokenizer(text, return_tensors="pt")
    outputs = deepmoji_model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    max_score_idx = torch.argmax(scores).item()
    return scores[0][max_score_idx].item()

def get_combined_sentiment_with_emojis(text):
    text_sentiment = get_combined_sentiment(text)
    emojis = extract_emojis(text)
    if emojis:
        emoji_sentiments = [get_deepmoji_sentiment(e) for e in emojis]
        combined_score = text_sentiment + sum(emoji_sentiments) / len(emoji_sentiments)
    else:
        combined_score = text_sentiment
    return combined_score

def combine_emotion_results(results):
    grouped_emotions = {
        'positive': ['admiration', 'approval', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'],
        'negative': ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'],
        'neutral': ['neutral'],
        'surprise': ['amusement', 'confusion', 'curiosity', 'excitement', 'realization', 'surprise'],
        'caring': ['caring'],
        'desire': ['desire'],
        'embarrassment': ['embarrassment']
    }
    
    combined_results = {}
    for label, score in results.items():
        for category, labels in grouped_emotions.items():
            if label in labels:
                if category not in combined_results:
                    combined_results[category] = []
                combined_results[category].append(score)
    
    averaged_results = {category: sum(scores) / len(scores) for category, scores in combined_results.items()}
    return averaged_results
