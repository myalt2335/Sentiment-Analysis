import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from afinn import Afinn
import torch
import emoji
from rich.console import Console
from rich.table import Table
from typing import Tuple, List

nltk.download('vader_lexicon')

console = Console()

sia = SentimentIntensityAnalyzer()
afinn = Afinn()
transformer_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
emotion_pipeline = pipeline(
    "sentiment-analysis",
    model="j-hartmann/emotion-english-distilroberta-base"
)
emotion_text_cnn_pipeline = pipeline(
    "sentiment-analysis",
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)
go_tokenizer = AutoTokenizer.from_pretrained(
    "monologg/bert-base-cased-goemotions-original"
)
go_model = AutoModelForSequenceClassification.from_pretrained(
    "monologg/bert-base-cased-goemotions-original"
)
deepmoji_tokenizer = AutoTokenizer.from_pretrained(
    'j-hartmann/emotion-english-distilroberta-base'
)
deepmoji_model = AutoModelForSequenceClassification.from_pretrained(
    'j-hartmann/emotion-english-distilroberta-base'
)

THRESHOLD = 0.1

def interpret_score(score: float) -> str:
    if score > THRESHOLD:
        return 'Positive'
    elif score < -THRESHOLD:
        return 'Negative'
    else:
        return 'Neutral'

def get_vader_sentiment(text: str) -> Tuple[str, float]:
    score = sia.polarity_scores(text)['compound']
    return interpret_score(score), score

def get_textblob_sentiment(text: str) -> Tuple[str, float]:
    score = TextBlob(text).sentiment.polarity
    return interpret_score(score), score

def get_transformer_sentiment(text: str) -> Tuple[str, float]:
    res = transformer_pipeline(text)[0]
    return res['label'], res['score']

def get_afinn_sentiment(text: str) -> Tuple[str, float]:
    score = afinn.score(text) / 10.0
    return interpret_score(score), score

def get_emotion_pipeline_sentiment(text: str) -> List[Tuple[str, float]]:
    e1 = emotion_pipeline(text)[0]
    e2 = emotion_text_cnn_pipeline(text)[0]
    return [(e1['label'], e1['score']), (e2['label'], e2['score'])]

def get_goemotions_sentiment(text: str) -> Tuple[str, float]:
    inputs = go_tokenizer(text, return_tensors='pt')
    outputs = go_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    labels = [
        'admiration','amusement','anger','annoyance','approval','caring','confusion',
        'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
        'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
        'pride','realization','relief','remorse','sadness','surprise','neutral'
    ]
    idx = torch.argmax(probs, dim=1).item()
    return labels[idx], probs[0][idx].item()

def get_deepmoji_sentiment(text: str) -> float:
    inputs = deepmoji_tokenizer(text, return_tensors='pt')
    outputs = deepmoji_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.max().item()

def get_combined_sentiment(text: str) -> float:
    _, vader_score = get_vader_sentiment(text)
    _, blob_score = get_textblob_sentiment(text)
    _, transformer_score = get_transformer_sentiment(text)
    _, afinn_score = get_afinn_sentiment(text)
    return vader_score * 0.45 + blob_score * 0.3 + transformer_score * 0.2 + afinn_score * 0.05

def analyze_and_display(text: str) -> None:
    overall_label, overall_score = (
        interpret_score(get_combined_sentiment(text)),
        get_combined_sentiment(text)
    )
    vader_label, vader_score = get_vader_sentiment(text)
    blob_label, blob_score = get_textblob_sentiment(text)
    transformer_label, transformer_score = get_transformer_sentiment(text)
    afinn_label, afinn_score = get_afinn_sentiment(text)
    emo_preds = get_emotion_pipeline_sentiment(text)
    go_label, go_score = get_goemotions_sentiment(text)
    deepmoji_score = get_deepmoji_sentiment(text)

    table = Table(title="Sentiment Analysis Results", header_style="bold cyan")
    table.add_column("Model", style="bold")
    table.add_column("Prediction")
    table.add_column("Score", justify="right")

    table.add_row("Overall", overall_label, f"{overall_score:.2f}")
    table.add_row("VADER", vader_label, f"{vader_score:.2f}")
    table.add_row("TextBlob", blob_label, f"{blob_score:.2f}")
    table.add_row("Transformer", transformer_label, f"{transformer_score:.2f}")
    table.add_row("Afinn", afinn_label, f"{afinn_score:.2f}")

    for idx, (label, score) in enumerate(emo_preds, start=1):
        table.add_row(f"Emotion{idx}", label, f"{score:.2f}")
    table.add_row("GoEmotions", go_label, f"{go_score:.2f}")
    table.add_row("DeepMoji", "", f"{deepmoji_score:.2f}")

    console.print(table)

def main() -> None:
    console.print("[bold underline]Sentiment Analysis[/bold underline]\n")
    while True:
        text = console.input("[bold green]Enter message (or 'exit'):[/bold green] ")
        if text.lower().strip() == 'exit':
            console.print("[bold red]Bye![/bold red]")
            break
        analyze_and_display(text)

if __name__ == '__main__':
    main()
