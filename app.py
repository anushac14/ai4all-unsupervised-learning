import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import numpy as np
from scipy.sparse import csr_matrix


# Import your custom classes and functions (assuming they are in another file, or you can include them directly here)
class ProductRecommender:
    def __init__(self, df, feature_matrix, cluster_labels):
        self.df = df
        self.feature_matrix = feature_matrix
        self.cluster_labels = cluster_labels

    def find_similar_products(self, product_name):
        # Find the product in the dataframe
        product_index = self.df[self.df['Product Name'].str.contains(product_name, case=False)].index
        if len(product_index) == 0:
            raise ValueError(f"Product '{product_name}' not found in the dataset.")
        
        # Get the feature vector for the product
        product_feature_vector = self.feature_matrix[product_index[0]]

        # Compute the similarity with all other products
        similarities = cosine_similarity(self.feature_matrix, product_feature_vector).flatten()

        # Add similarities to the dataframe
        self.df['Similarity'] = similarities
        
        # Sort by similarity and return top N recommendations (e.g., 5)
        top_recommendations = self.df.sort_values(by='Similarity', ascending=False).head(5)
        return top_recommendations



# Preprocess and helper functions
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

    processed_df['Selling Price'] = pd.to_numeric(
        processed_df['Selling Price'].astype('str').str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    )

    processed_df['Shipping Weight'] = processed_df['Shipping Weight'].apply(to_ounces)

    imputer = SimpleImputer(strategy='median')
    processed_df['Selling Price'] = imputer.fit_transform(processed_df[['Selling Price']])
    processed_df['Shipping Weight'] = imputer.fit_transform(processed_df[['Shipping Weight']])

    processed_df['Technical Details'] = processed_df['Technical Details'].fillna('No detail available')
    processed_df = processed_df.dropna(subset=['Category', 'Product Specification'])

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

    scaler = MinMaxScaler()
    numeric_features = scaler.fit_transform(
        processed_df[['Selling Price', 'Shipping Weight']]
    )

    # Use hstack to combine tfidf and numeric features
    feature_matrix = hstack([tfidf_matrix, numeric_features])

    # Convert the sparse matrix to csr_matrix for efficient row indexing
    return csr_matrix(feature_matrix)

def create_recommender(df, feature_matrix, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    return ProductRecommender(df, feature_matrix, cluster_labels)

# Streamlit app
def main():
    st.title("Product Recommender")

    # Load the data
    df = pd.read_csv('data.csv')

    # Preprocess the data
    processed_df = preprocess_data(df)

    # Create feature matrix
    feature_matrix = create_feature_matrix(processed_df)

    # Create the recommender system
    recommender = create_recommender(processed_df, feature_matrix)

    # User input for product search
    product_name = st.text_input("Enter a product name:")

    if st.button("Recommend Products"):
        if product_name:
            try:
                recommendations = recommender.find_similar_products(product_name)
                st.write(f"Top recommendations for '{product_name}':")
                st.dataframe(recommendations[['Product Name', 'Category', 'Selling Price', 'Similarity']])
            except ValueError as e:
                st.error(e)
        else:
            st.warning("Please enter a product name.")

if __name__ == "__main__":
    main()
