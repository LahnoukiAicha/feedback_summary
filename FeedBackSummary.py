# app.py
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
from summarizer import Summarizer

# Init Flask app
app = Flask(__name__)

# Load model/tokenizer once
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
summarizer = Summarizer(custom_model=model, custom_tokenizer=tokenizer)

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    document = data.get('document', '')

    if not document.strip():
        return jsonify({'error': 'No document content provided'}), 400

    try:
        summary = summarizer(document, ratio=0.1, num_sentences=3, use_first=False)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
