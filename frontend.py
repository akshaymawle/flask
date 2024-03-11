from flask import Flask, render_template, request, jsonify
from backend import read_pdf, calculate_cosine_similarity

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1 and file2:
        text1 = read_pdf(file1)
        text2 = read_pdf(file2)
        similarity_score = calculate_cosine_similarity(text1, text2)
        return jsonify({'similarity_score': similarity_score})
    else:
        return jsonify({'error': 'Please upload both PDF files.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
