from surprise import KNNWithMeans, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('ratings.csv', header=None, names=['userId', 'productId', 'Rating', 'timestamp'])
df = df.drop(columns=['timestamp']) 

df_sampled = df.sample(n=500000, random_state=10) 

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_sampled[['userId', 'productId', 'Rating']], reader)

trainset, testset = train_test_split(data, test_size=0.3, random_state=10)

algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})

algo.fit(trainset)

test_pred = algo.test(testset)
print("Item-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)

def get_recommendations_for_user(user_id, algo, num_recommendations=5):
    items_user_has_not_rated = [iid for iid in df_sampled['productId'].unique() if not trainset.knows_item(iid)]

    predictions = [algo.predict(user_id, iid) for iid in items_user_has_not_rated]

    predictions.sort(key=lambda x: x.est, reverse=True)

    recommended_items = [(pred.iid, pred.est) for pred in predictions[:num_recommendations]]
    return recommended_items

user_id = 'A2CX7LUOHB2NDG'  
recommendations = get_recommendations_for_user(user_id, algo, num_recommendations=5)
print(f"Top recommendations for User {user_id}:", recommendations)
