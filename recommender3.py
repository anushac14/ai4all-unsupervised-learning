import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

products = pd.read_csv('data.csv', on_bad_lines='skip')

products = products[['Product Name', 'Brand Name', 'Product Description', 'Product Details', 'Category']]  # Added Category

products['Product Description'] = products['Product Description'].fillna('')
products['Product Details'] = products['Product Details'].fillna('')
products['Brand Name'] = products['Brand Name'].fillna('')
products['Category'] = products['Category'].fillna('') 

with open('stopwords.txt', 'r') as file:
    stop_words = list(file.read().splitlines()) 

def clean_text(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.lower().translate(translator)
    words = [word for word in text.split() if word not in stop_words]
    
    if not words: 
        return "emptytext"
    return ' '.join(words)

products['cleaned_description'] = products['Product Description'].apply(clean_text)
products['cleaned_details'] = products['Product Details'].apply(clean_text)

products['combined_features'] = (
    products['Brand Name'] + ' ' + 
    products['cleaned_description'] + ' ' + 
    products['cleaned_details'] + ' ' +
    products['Category']
)

tfidf = TfidfVectorizer(stop_words=stop_words) 

tfidf_matrix = tfidf.fit_transform(products['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

products = products.reset_index(drop=True)
indices = pd.Series(products.index, index=products['Product Name']).drop_duplicates()

def recommend_products_by_input(user_input, cosine_sim=cosine_sim, products=products, indices=indices, num_recommendations=5):
    tfidf_input = tfidf.transform([user_input])

    input_sim_scores = cosine_similarity(tfidf_input, tfidf_matrix).flatten()

    sim_scores = list(enumerate(input_sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[:num_recommendations]
    product_indices = [i[0] for i in sim_scores]

    return products['Product Name'].iloc[product_indices]

user_input = "Bicycles" 
print(f"\nRecommendations for '{user_input}':")
recommendations = recommend_products_by_input(user_input)

for i, rec in enumerate(recommendations, start=1):
    print(f"{i}. {rec}")
