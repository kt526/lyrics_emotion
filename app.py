from flask import Flask, render_template, request, jsonify

from flask_cors import CORS
from predict import *

app = Flask(__name__)   
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/process_lyrics', methods=['POST'])
def process_lyrics():
    data = request.get_json()

    # Extract model_k and lyrics_k from the received JSON data
    raw_lyrics = data.get('lyrics')

    # Perform processing or analysis here (replace this with your actual processing logic)
    print(f"Processing lyrics using model: {raw_lyrics}")

    emotion = predict_emotion(raw_lyrics)

    # load the model from disk
    # Return the processed result as JSON
    return jsonify({'data': emotion})

@app.route('/test', methods=['GET'])
def test():
  return 'test'

if __name__ == "__main__":
    app.run(debug=True)
