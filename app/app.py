from flask import Flask, request, jsonify
from utils import load_data, preprocess_data, create_user_item_matrix
from model import train_svd, recommend_svd
import os

app = Flask(__name__)

# Load and preprocess data
data_path = os.path.join('data', 'ratings.csv')
ratings = load_data(data_path)
ratings = preprocess_data(ratings)
user_item_matrix = create_user_item_matrix(ratings)

# Train SVD model
svd, _ = train_svd(user_item_matrix)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']
    recommendations = recommend_svd(svd, user_item_matrix, user_id)
    return jsonify(recommendations.tolist())

if __name__ == '__main__':
    app.run(debug=True)
