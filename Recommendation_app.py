import streamlit as st
import pandas as pd
import numpy as np
from model_persistence import load_recommendation_model
import plotly.express as px
import plotly.graph_objects as go
import os
from product_recommender import ProductRecommender

class StreamlitRecommenderApp:
    def __init__(self, model_path):
        """Initialize the Streamlit app with the recommendation model"""
        # Set page config
        st.set_page_config(
            page_title="Product Recommender",
            page_icon="🛍️",
            layout="wide"
        )
        
        # Load model
        try:
            self.recommender = load_recommendation_model(model_path)
            self.df = self.recommender.df
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    def create_cluster_visualization(self):
        """Create visualization of product clusters"""
        cluster_stats = self.df.groupby('cluster').agg({
            'Product Name': 'count',
            'Selling Price': ['mean', 'min', 'max'],
            'Category': lambda x: ', '.join(x.value_counts().head(3).index)
        }).round(2)
        
        # Prepare data for visualization
        cluster_sizes = cluster_stats['Product Name']['count']
        avg_prices = cluster_stats['Selling Price']['mean']
        
        # Create bubble chart
        fig = px.scatter(
            x=range(len(cluster_sizes)),
            y=avg_prices,
            size=cluster_sizes,
            title="Cluster Overview",
            labels={
                'x': 'Cluster ID',
                'y': 'Average Price ($)',
                'size': 'Number of Products'
            }
        )
        
        return fig, cluster_stats
    
    def show_product_details(self, product_name):
        """Display detailed information about a product"""
        product = self.df[self.df['Product Name'] == product_name.lower()].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Product Details")
            st.write(f"**Category:** {product['Category']}")
            st.write(f"**Price:** ${product['Selling Price']:.2f}")
            st.write(f"**Cluster:** {product['cluster']}")
        
        with col2:
            # Show cluster statistics
            cluster_products = self.df[self.df['cluster'] == product['cluster']]
            st.subheader("Cluster Statistics")
            st.write(f"**Total Products in Cluster:** {len(cluster_products)}")
            st.write(f"**Average Price:** ${cluster_products['Selling Price'].mean():.2f}")
            st.write(f"**Price Range:** ${cluster_products['Selling Price'].min():.2f} - "
                    f"${cluster_products['Selling Price'].max():.2f}")
    
    def show_recommendations(self, product_name, n_recommendations):
        """Display recommendations for a product"""
        try:
            recommendations = self.recommender.find_similar_products(
                product_name, 
                n_recommendations
            )
            
            # Create similarity score visualization
            fig = go.Figure(go.Bar(
                x=recommendations['Product Name'],
                y=recommendations['Similarity'],
                text=recommendations['Similarity'].round(3),
                textposition='auto',
            ))
            fig.update_layout(
                title="Similarity Scores",
                xaxis_title="Product",
                yaxis_title="Similarity Score",
                showlegend=False
            )
            
            # Display recommendations
            st.plotly_chart(fig)
            
            # Display recommendations in a table
            st.write("### Recommended Products")
            for _, row in recommendations.iterrows():
                with st.expander(f"{row['Product Name']} (${row['Selling Price']:.2f})"):
                    st.write(f"**Category:** {row['Category']}")
                    st.write(f"**Similarity Score:** {row['Similarity']:.3f}")
                    st.write(f"**Price:** ${row['Selling Price']:.2f}")
        
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
    
    def run(self):
        """Run the Streamlit app"""
        st.title("🛍️ Product Recommendation System")
        
        # Sidebar
        st.sidebar.header("Settings")
        
        # Product selection
        product_names = sorted(self.df['Product Name'].str.title().unique())
        selected_product = st.sidebar.selectbox(
            "Select a Product",
            options=product_names
        )
        
        # Number of recommendations
        n_recommendations = st.sidebar.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=20,
            value=5
        )
        
        # Main content
        if selected_product:
            self.show_product_details(selected_product)
            
            # Show recommendations
            st.header("Product Recommendations")
            self.show_recommendations(selected_product, n_recommendations)
            
            # Show cluster visualization
            st.header("Cluster Analysis")
            fig, cluster_stats = self.create_cluster_visualization()
            st.plotly_chart(fig)
            
            # Show cluster statistics
            with st.expander("View Detailed Cluster Statistics"):
                st.dataframe(cluster_stats)

def main():
    # Model path - update this with your model path
    model_path = "saved_models/product_recommender_v1"  # Update this path
    
    # Create and run app
    app = StreamlitRecommenderApp(model_path)
    app.run()

if __name__ == "__main__":
    main()