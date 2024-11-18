import joblib
import os
import json
from datetime import datetime
from scipy.sparse import save_npz, load_npz

class ModelPersistence:
    def __init__(self, base_path='saved_models'):
        
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_model(self, recommender, model_name=None):
        
        # Generate model directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_name or f"model_{timestamp}"
        model_dir = os.path.join(self.base_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Save feature matrix
            save_npz(os.path.join(model_dir, 'feature_matrix.npz'), 
                    recommender.feature_matrix)
            
            # Save DataFrame
            recommender.df.to_pickle(os.path.join(model_dir, 'processed_df.pkl'))
            
            # Save product_to_idx mapping
            with open(os.path.join(model_dir, 'product_to_idx.json'), 'w') as f:
                json.dump(recommender.product_to_idx, f)
            
            # Save model metadata
            metadata = {
                'timestamp': timestamp,
                'n_products': len(recommender.df),
                'n_features': recommender.feature_matrix.shape[1],
                'n_clusters': len(recommender.df['cluster'].unique()),
                'model_name': model_name
            }
            with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            print(f"Model successfully saved to {model_dir}")
            return model_dir
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path):
        
        try:
            # Load feature matrix
            feature_matrix = load_npz(os.path.join(model_path, 'feature_matrix.npz'))
            
            # Load DataFrame
            df = pd.read_pickle(os.path.join(model_path, 'processed_df.pkl'))
            
            # Load product_to_idx mapping
            with open(os.path.join(model_path, 'product_to_idx.json'), 'r') as f:
                product_to_idx = json.load(f)
            
            # Create recommender instance
            recommender = ProductRecommender(df, feature_matrix, df['cluster'].values)
            recommender.product_to_idx = product_to_idx
            
            # Load and print metadata
            with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
            print("\nLoaded model information:")
            for key, value in metadata.items():
                print(f"{key}: {value}")
            
            return recommender
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def list_saved_models(self):
        
        models = []
        for model_dir in os.listdir(self.base_path):
            metadata_path = os.path.join(self.base_path, model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                models.append(metadata)
        
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)

# Example usage functions
def save_recommendation_model(recommender, model_name=None):
    
    persistence = ModelPersistence()
    return persistence.save_model(recommender, model_name)

def load_recommendation_model(model_path):
   
    persistence = ModelPersistence()
    return persistence.load_model(model_path)

def list_available_models():
    
    persistence = ModelPersistence()
    models = persistence.list_saved_models()
    
    print("\nAvailable Models:")
    print("-" * 80)
    for model in models:
        print(f"\nModel Name: {model['model_name']}")
        print(f"Saved: {model['timestamp']}")
        print(f"Products: {model['n_products']}")
        print(f"Features: {model['n_features']}")
        print(f"Clusters: {model['n_clusters']}")
        print("-" * 40)
