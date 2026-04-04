"""
Restaurant Recommendation System - Model Training
Uses Content-Based Filtering with TF-IDF Vectorizer
Dataset: Zomato Bangalore Restaurants
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import warnings
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import os

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

# ============================================================
# 1. DATA COLLECTION & LOADING
# ============================================================
print("=" * 60)
print("RESTAURANT RECOMMENDATION SYSTEM - MODEL TRAINING")
print("=" * 60)

# Load dataset
dataset_path = '../Dataset/zomato.csv'
if not os.path.exists(dataset_path):
    # Try alternative paths
    for path in ['zomato.csv', 'Dataset/zomato.csv', '../zomato.csv']:
        if os.path.exists(path):
            dataset_path = path
            break

print(f"\n[1] Loading dataset from: {dataset_path}")

try:
    df = pd.read_csv(dataset_path, encoding='latin-1')
    print(f"    Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("    Dataset not found! Creating sample dataset for demonstration...")
    # Create a sample dataset
    sample_data = {
        'name': ['Jalsa', 'Cinnamon', 'Biergarten', 'The Pallet', 'Delhi Highway',
                 'Crawl Street', 'Eggzotic', 'Madeena Hotel', 'Pallavi Restaurant',
                 'Desi Doze', 'Insomniac\'s Delight', 'Donne Biriyani Angadi Mane',
                 'Agarwal Food Service', 'Hotel New Karavali', 'Atithi',
                 'Deja Vu Resto Bar', 'The Fisherman\'s Wharf', 'West Wood',
                 'Punjab Grill', 'Meghana Foods', 'Empire Restaurant', 'CTR',
                 'Vidyarthi Bhavan', 'MTR', 'Koshy\'s', 'The Black Pearl',
                 'Toit', 'Arbor Brewing Company', 'Biere Club', 'Windmills Craftworks'],
        'cuisines': [
            'North Indian, Chinese, Continental',
            'North Indian, Asian, Continental',
            'Continental, North Indian, Chinese, European, BBQ, Finger Food, Asian',
            'Continental, Mediterranean, Italian, North Indian, Finger Food, Asian, Momos',
            'North Indian, Mughlai',
            'Continental, Finger Food, North Indian, Chinese',
            'North Indian, Chinese, Biryani, Fast Food',
            'North Indian, Mughlai, Biryani',
            'Biryani, Chinese, Andhra',
            'North Indian, Fast Food',
            'Fast Food, North Indian',
            'Biryani, Chinese',
            'North Indian, Chinese, Biryani',
            'Mangalorean, South Indian, North Indian',
            'North Indian, Chinese, Street Food',
            'North Indian, Italian',
            'Seafood, Goan, North Indian, Continental, Asian',
            'North Indian, Chinese, Continental',
            'North Indian, Mughlai, Biryani',
            'Biryani, Andhra',
            'North Indian, Chinese, Fast Food',
            'South Indian',
            'South Indian',
            'South Indian',
            'Continental, North Indian',
            'Seafood, Continental, Goan',
            'Continental, Finger Food, Craft Beer',
            'Continental, Craft Beer, American',
            'Continental, Craft Beer, European',
            'Continental, Craft Beer, American'
        ],
        'rate': [4.1, 3.6, 4.8, 4.5, 4.4, 4.2, 3.8, 3.8, 3.6, 3.6,
                 3.8, 3.5, 3.4, 3.3, 3.6, 4.4, 4.3, 3.5, 4.0, 4.4,
                 3.9, 4.5, 4.5, 4.5, 4.2, 4.1, 4.4, 4.3, 4.2, 4.1],
        'votes': [775, 2924, 821, 2182, 3024, 1842, 573, 482, 283, 421,
                  312, 195, 287, 341, 432, 1823, 2341, 234, 1234, 3421,
                  2341, 1823, 2145, 3421, 1234, 823, 2341, 1823, 2145, 1234],
        'approx_cost(for two people)': [800, 800, 1400, 1200, 1000, 800, 500, 400, 500, 400,
                                         300, 250, 400, 300, 800, 900, 1200, 800, 1200, 400,
                                         400, 200, 100, 200, 400, 1200, 1200, 1200, 1200, 1200],
        'rest_type': ['Casual Dining', 'Casual Dining', 'Casual Dining', 'Casual Dining', 'Casual Dining',
                      'Quick Bites', 'Quick Bites', 'Quick Bites', 'Casual Dining', 'Quick Bites',
                      'Quick Bites', 'Quick Bites', 'Quick Bites', 'Casual Dining', 'Quick Bites',
                      'Casual Dining', 'Casual Dining', 'Casual Dining', 'Fine Dining', 'Casual Dining',
                      'Quick Bites', 'Quick Bites', 'Cafe', 'Casual Dining', 'Cafe',
                      'Casual Dining', 'Casual Dining', 'Bar', 'Bar', 'Bar'],
        'location': ['Banashankari', 'Bellandur', 'Sankey Road', 'Koramangala 7th Block', 'Indiranagar',
                     'Indiranagar', 'Koramangala 5th Block', 'Shivajinagar', 'Koramangala', 'Koramangala',
                     'Koramangala', 'Jayanagar', 'Jayanagar', 'Indiranagar', 'HSR',
                     'Indiranagar', 'Indiranagar', 'Indiranagar', 'Indiranagar', 'Koramangala',
                     'Residency Road', 'Malleshwaram', 'Basavanagudi', 'Jayanagar', 'St. Marks Road',
                     'Indiranagar', 'Indiranagar', 'Indiranagar', 'Lavelle Road', 'Whitefield']
    }
    df = pd.DataFrame(sample_data)
    os.makedirs('../Dataset', exist_ok=True)
    df.to_csv(dataset_path, index=False)
    print(f"    Sample dataset created with {len(df)} restaurants")

# ============================================================
# 2. DATA PRE-PROCESSING
# ============================================================
print("\n[2] Data Pre-processing...")

# Check columns
print(f"    Columns: {list(df.columns)}")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# Rename cost column if it exists
cost_col = [c for c in df.columns if 'cost' in c or 'approx' in c]
if cost_col:
    df.rename(columns={cost_col[0]: 'cost'}, inplace=True)

# Handle rate column
if 'rate' in df.columns:
    df['rate'] = df['rate'].astype(str)
    df['rate'] = df['rate'].apply(lambda x: x.replace('/5', '').strip() if '/5' in str(x) else x)
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Handle votes
if 'votes' in df.columns:
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

# Drop duplicates
df.drop_duplicates(inplace=True)
print(f"    After dedup: {df.shape[0]} rows")

# Fill missing values
df['cuisines'] = df['cuisines'].fillna('Unknown')
df['rate'] = df['rate'].fillna(df['rate'].median() if 'rate' in df.columns else 3.5)
df['votes'] = df['votes'].fillna(0)

# Handle cost column
if 'cost' in df.columns:
    df['cost'] = df['cost'].astype(str).str.replace(',', '').str.strip()
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    df['cost'] = df['cost'].fillna(df['cost'].median())
    # Convert to thousands
    df['cost'] = (df['cost'] / 1000).round(1)
else:
    df['cost'] = 0.5

# ============================================================
# 3. COMPUTE MEAN RATING
# ============================================================
print("\n[3] Computing Mean Ratings...")

# Group by restaurant name and compute mean rating
rating_df = df.groupby('name').agg({'rate': 'mean', 'votes': 'sum'}).reset_index()
rating_df.columns = ['name', 'mean_rating', 'total_votes']
rating_df['mean_rating'] = rating_df['mean_rating'].round(2)

# Merge back
df = df.merge(rating_df, on='name', how='left')

# ============================================================
# 4. EXPLORATORY DATA ANALYSIS & VISUALIZATION
# ============================================================
print("\n[4] Generating Visualizations...")

os.makedirs('../static/plots', exist_ok=True)

try:
    # Top cuisines
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Restaurant Data Analysis - Bangalore', fontsize=16, fontweight='bold')

    # Plot 1: Top 15 Most Common Cuisines
    all_cuisines = []
    for c in df['cuisines'].dropna():
        all_cuisines.extend([x.strip() for x in str(c).split(',')])
    cuisine_counts = pd.Series(all_cuisines).value_counts().head(15)
    
    axes[0, 0].barh(cuisine_counts.index[::-1], cuisine_counts.values[::-1], 
                     color=plt.cm.viridis(np.linspace(0.2, 0.8, 15)))
    axes[0, 0].set_title('Top 15 Most Popular Cuisines', fontweight='bold')
    axes[0, 0].set_xlabel('Count')

    # Plot 2: Rating Distribution
    axes[0, 1].hist(df['rate'].dropna(), bins=20, color='#e67e22', edgecolor='white', alpha=0.8)
    axes[0, 1].set_title('Restaurant Rating Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Rating')
    axes[0, 1].set_ylabel('Count')

    # Plot 3: Cost Distribution
    axes[1, 0].hist(df['cost'].dropna(), bins=20, color='#2ecc71', edgecolor='white', alpha=0.8)
    axes[1, 0].set_title('Cost Distribution (in Thousands)', fontweight='bold')
    axes[1, 0].set_xlabel('Cost (₹ thousands)')
    axes[1, 0].set_ylabel('Count')

    # Plot 4: Rating vs Cost scatter
    axes[1, 1].scatter(df['cost'].dropna(), df['rate'].dropna(), 
                        alpha=0.4, color='#9b59b6', s=20)
    axes[1, 1].set_title('Rating vs Cost', fontweight='bold')
    axes[1, 1].set_xlabel('Cost (₹ thousands)')
    axes[1, 1].set_ylabel('Rating')

    plt.tight_layout()
    plt.savefig('../static/plots/eda_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("    EDA plots saved.")
except Exception as e:
    print(f"    Visualization skipped: {e}")

# ============================================================
# 5. CONTENT-BASED FILTERING
# ============================================================
print("\n[5] Building Content-Based Filtering Model...")

# Create a unique restaurants dataframe (one entry per restaurant)
restaurant_df = df.drop_duplicates(subset='name').copy()
restaurant_df = restaurant_df.reset_index(drop=True)

print(f"    Unique restaurants: {len(restaurant_df)}")

# Create TF-IDF matrix from cuisines
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), 
                         min_df=0.0, stop_words='english')

# Clean cuisines text
restaurant_df['cuisines_clean'] = restaurant_df['cuisines'].fillna('').str.lower().str.replace(',', ' ')

tfidf_matrix = tfidf.fit_transform(restaurant_df['cuisines_clean'])
print(f"    TF-IDF matrix shape: {tfidf_matrix.shape}")

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(f"    Cosine similarity matrix shape: {cosine_sim.shape}")

# Create reverse index (name -> index)
indices = pd.Series(restaurant_df.index, index=restaurant_df['name']).drop_duplicates()

# ============================================================
# 6. SAVE MODEL
# ============================================================
print("\n[6] Saving Model...")

model_data = {
    'df': restaurant_df,
    'cosine_sim': cosine_sim,
    'indices': indices,
    'tfidf': tfidf
}

with open('restaurant.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Also save processed CSV
restaurant_df.to_csv('restaurant1.csv', index=False)

print("    Model saved: restaurant.pkl")
print("    Data saved: restaurant1.csv")

# ============================================================
# 7. QUICK VALIDATION
# ============================================================
print("\n[7] Validating Model...")

def get_recommendations(name, n=10):
    try:
        idx = indices[name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        rest_indices = [i[0] for i in sim_scores]
        result = restaurant_df[['name', 'cuisines', 'mean_rating', 'cost']].iloc[rest_indices]
        return result
    except KeyError:
        return None

# Test with first restaurant
test_name = restaurant_df['name'].iloc[0]
recs = get_recommendations(test_name)
if recs is not None:
    print(f"\n    Sample recommendations for '{test_name}':")
    print(recs[['name', 'cuisines', 'mean_rating']].to_string(index=False))

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("Run: python app1.py")
print("=" * 60)
