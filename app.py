# app.py
from flask import Flask, request, jsonify
from transformers import pipeline
from langdetect import detect
import emoji

app = Flask(__name__)

# Sentiment analizini gerçekleştirecek pipeline oluşturuluyor
sentiment_analysis = pipeline("sentiment-analysis")

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get('text', '')
    
    # Metindeki emojileri unicode olarak tanımla
    text_with_emojis = emoji.demojize(text)
    
    # Dil tespit et
    language = detect(text)
    
    # Sentiment analizi yap
    result = sentiment_analysis(text_with_emojis)
    
    # Sonuçları döndür
    return jsonify({
        'sentiment': result[0]['label'],
        'score': result[0]['score'],
        'language': language
    })

if __name__ == '__main__':
    app.run(debug=True)
