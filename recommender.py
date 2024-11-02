import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('ratings.csv', header=None, names=['user_id', 'product_id', 'rating', 'timestamp'])

df = df.drop(columns=['timestamp'])

df_first_10 = df.head(100)

user_item_matrix = df_first_10.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
user_item_sparse = csr_matrix(user_item_matrix.values)

item_similarity_matrix = cosine_similarity(user_item_sparse.T)

item_similarity_df = pd.DataFrame(item_similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def recommend_items(user_id, user_item_matrix, item_similarity_df, num_recommendations=5):
    user_ratings = user_item_matrix.loc[user_id]

    rated_items = user_ratings[user_ratings > 0].index.tolist()

    item_scores = pd.Series(dtype='float64')

    for item in rated_items:
        similarity_scores = item_similarity_df[item]

        weighted_scores = similarity_scores * user_ratings[item]

        item_scores = item_scores.add(weighted_scores, fill_value=0)

    item_scores = item_scores.drop(rated_items, errors='ignore')

    recommended_items = item_scores.sort_values(ascending=False).head(num_recommendations)

    return recommended_items

user_id = user_item_matrix.index[0]  
recommendations = recommend_items(user_id, user_item_matrix, item_similarity_df, num_recommendations=5)
print(f"Top recommendations for User {user_id}:\n", recommendations)
