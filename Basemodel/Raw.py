import os
import subprocess
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

sia = SentimentIntensityAnalyzer()
transformer_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
afinn = Afinn()
flair_sentiment = TextClassifier.load('sentiment-fast')
emotion_pipeline = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")
emotion_text_cnn_pipeline = pipeline("sentiment-analysis", model="bhadresh-savani/distilbert-base-uncased-emotion")
sarcasm_model = pipeline("text-classification", model="nikesh66/Sarcasm-Detection-using-BERT")

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
    sentence = Sentence(text)
    flair_sentiment.predict(sentence)
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

def detect_sarcasm(text):
    results = sarcasm_model(text)
    for result in results:
        if result['label'] == "Sarcasm":
            return True, result['score']
    return False, 0.0

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

def analyze_text(text):
    combined_score = get_combined_sentiment_with_emojis(text)
    
    if combined_score > 0.05:
        basic_sentiment_category = 'positive'
    elif combined_score < -0.05:
        basic_sentiment_category = 'negative'
    else:
        basic_sentiment_category = 'neutral'

    emotion = get_emotion_sentiment(text)
    emotion_text_cnn = get_emotion_text_cnn_sentiment(text)
    goemotion_label, goemotion_score = get_goemotions_sentiment(text)

    emotion_results = {
        emotion['label']: emotion['score'],
        emotion_text_cnn['label']: emotion_text_cnn['score'],
        goemotion_label: goemotion_score
    }

    combined_emotions = {}
    for label, score in emotion_results.items():
        if label in combined_emotions:
            combined_emotions[label].append(score)
        else:
            combined_emotions[label] = [score]

    for label, scores in combined_emotions.items():
        combined_emotions[label] = sum(scores) / len(scores)

    response = f'The text is {basic_sentiment_category}. '
    for label, avg_score in combined_emotions.items():
        response += f'It was also detected as {label} with a confidence of {avg_score:.2f}. '

    sarcasm_detected, sarcasm_score = detect_sarcasm(text)
    if sarcasm_detected:
        response += f'Sarcasm detected with a confidence of {sarcasm_score:.2f}.'

    return response
