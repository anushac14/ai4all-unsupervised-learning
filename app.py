import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix

df = pd.read_csv('data.csv')  # Load your actual dataset here

def to_ounces(weight):
    if pd.isna(weight):
        return np.nan
    weight_str = str(weight).strip().lower()
    try:
        if 'pound' in weight_str:
            return float(weight_str.split()[0]) * 16
        elif 'ounce' in weight_str:
            return float(weight_str.split()[0])
        else:
            return np.nan
    except ValueError:
        return np.nan

def preprocess_data(df):
    processed_df = df[['Product Name', 'Category', 'Selling Price',
                      'Product Specification', 'Technical Details',
                      'Shipping Weight', 'Is Amazon Seller']].copy()

    # Clean Selling Price
    processed_df['Selling Price'] = pd.to_numeric(
        processed_df['Selling Price'].astype('str').str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    )

    # Clean Shipping Weight
    processed_df['Shipping Weight'] = processed_df['Shipping Weight'].apply(to_ounces)

    # Impute numerical values
    imputer = SimpleImputer(strategy='median')
    processed_df['Selling Price'] = imputer.fit_transform(processed_df[['Selling Price']])
    processed_df['Shipping Weight'] = imputer.fit_transform(processed_df[['Shipping Weight']])

    # Handle text fields
    processed_df['Technical Details'] = processed_df['Technical Details'].fillna('No detail available')
    processed_df = processed_df.dropna(subset=['Category', 'Product Specification'])

    # Create combined text field
    processed_df['combined'] = (processed_df['Product Specification'] + " " +
                              processed_df['Technical Details'] + " " +
                              processed_df['Category'])

    return processed_df.reset_index(drop=True)

def create_feature_matrix(processed_df, max_features=10000, ngram_range=(1, 2)):
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        ngram_range=ngram_range
    )
    tfidf_matrix = tfidf.fit_transform(processed_df['combined'])

    # Scale numerical features
    scaler = MinMaxScaler()
    numeric_features = scaler.fit_transform(
        processed_df[['Selling Price', 'Shipping Weight']]
    )

    return hstack([tfidf_matrix, numeric_features])

class ProductRecommender:
    def __init__(self, df, feature_matrix, cluster_labels):
        self.df = df.copy()
        self.feature_matrix = csr_matrix(feature_matrix)
        self.df['cluster'] = cluster_labels

        # Normalize product names
        self.df['Product Name'] = self.df['Product Name'].str.strip().str.lower()

        # Create product name to index mapping
        self.product_to_idx = dict(zip(self.df['Product Name'], self.df.index))

    def find_similar_products(self, product_name: str, n_recommendations: int = 10) -> pd.DataFrame:
        try:
            # Normalize input product name
            product_name = product_name.strip().lower()

            # Check if product exists
            if product_name not in self.product_to_idx:
                raise ValueError(f"Product '{product_name}' not found in the dataset.")

            # Get product index and cluster
            idx = self.product_to_idx[product_name]
            product_cluster = self.df.loc[idx, 'cluster']

            # Filter products in the same cluster
            cluster_mask = self.df['cluster'] == product_cluster
            cluster_indices = self.df[cluster_mask].index

            # Calculate similarity scores
            product_features = self.feature_matrix[idx]
            cluster_features = self.feature_matrix[cluster_indices]
            sim_scores = cosine_similarity(product_features, cluster_features).flatten()

            # Create recommendations dataframe
            recommendations = pd.DataFrame({
                'Product Name': self.df.loc[cluster_indices, 'Product Name'],
                'Category': self.df.loc[cluster_indices, 'Category'],
                'Selling Price': self.df.loc[cluster_indices, 'Selling Price'],
                'Similarity': sim_scores,
                'Cluster': product_cluster
            })

            recommendations = (recommendations[recommendations.index != idx]
                            .sort_values('Similarity', ascending=False)
                            .head(n_recommendations)
                            .round({'Similarity': 3}))

            return recommendations

        except Exception as e:
            raise Exception(f"Error generating recommendations: {str(e)}")

def create_recommender(df, feature_matrix, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    return ProductRecommender(df, feature_matrix, cluster_labels)

# Main App
st.title('Product Recommendation System')

# Preprocess the data
processed_df = preprocess_data(df)

# Create feature matrix
feature_matrix = create_feature_matrix(processed_df)

# Initialize recommender
recommender = create_recommender(processed_df, feature_matrix)

# User input for product search
product_name = st.text_input('Enter the product name:')
n_recommendations = st.slider('Number of recommendations', 1, 20, 5)

# Button to trigger recommendations
if st.button('Get Recommendations'):
    if product_name:
        try:
            # Get recommendations
            recommendations = recommender.find_similar_products(product_name, n_recommendations)
            
            # Display the recommendations
            st.write(f"Recommendations for: **{product_name}**")
            st.write(recommendations[['Product Name', 'Category', 'Selling Price', 'Similarity']])

        except ValueError as e:
            st.error(str(e))
    else:
        st.error('Please enter a valid product name.')