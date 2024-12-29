from sklearn.decomposition import TruncatedSVD

def train_svd(matrix, n_components=50):
    svd = TruncatedSVD(n_components=n_components)
    matrix_transformed = svd.fit_transform(matrix)
    return svd, matrix_transformed

def recommend_svd(svd, matrix, user_id):
    predicted_ratings = svd.inverse_transform(matrix)
    user_ratings = predicted_ratings[user_id - 1]  # Adjust for 0-indexing
    return user_ratings
