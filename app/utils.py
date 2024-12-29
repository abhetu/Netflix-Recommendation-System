import pandas as pd

def load_data(filepath):
    ratings = pd.read_csv(filepath)
    return ratings

def preprocess_data(ratings):
    ratings = ratings.dropna()
    return ratings

def create_user_item_matrix(ratings):
    matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
    matrix.fillna(0, inplace=True)
    return matrix
