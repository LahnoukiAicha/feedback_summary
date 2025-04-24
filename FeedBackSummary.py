from flask import Flask, request,jsonify
from transformers import BertTokenizer, BertModel
from summarizer import Summarizer
from bs4 import BeautifulSoup
import re
import os
os.environ['CURL_CA_BUNDLE'] = ''

app = Flask(__name__)
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
summarizer = Summarizer(custom_model=model, custom_tokenizer=tokenizer)

# Cleaning function to remove excessive spaces and unwanted line breaks
def clean_text_pipeline(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    html_document = data.get('document', '')
    
    # Paramètres supplémentaires pour personnaliser le résumé
    ratio = float(data.get('ratio', 0.2))  # Par défaut à 0.15
    num_sentences = int(data.get('num_sentences', 3))  # Par défaut à 3 phrases
    use_first = bool(data.get('use_first', True))  # Par défaut à True pour garder la première phrase

    print("==== Received html document ====")
    print(html_document)
    print("==========================")

    # ✅ Remove HTML tags
    soup = BeautifulSoup(html_document, 'html.parser')
    clean_text = clean_text_pipeline(soup.get_text())
    
    print("-----Clean text -----")
    print(clean_text)
    
    if not clean_text.strip():
        return jsonify({'error': 'No text after HTML cleanup'}), 400

    try:
        # Résumer le texte
        summary = summarizer(
            clean_text,
            ratio=ratio,
            num_sentences=num_sentences,
            use_first=use_first
        )
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
