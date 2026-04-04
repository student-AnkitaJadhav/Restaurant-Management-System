from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model and data
try:
    with open('restaurant.pkl', 'rb') as f:
        data = pickle.load(f)
    df = data['df']
    cosine_sim = data['cosine_sim']
    indices = data['indices']
    MODEL_LOADED = True
except Exception as e:
    print(f"Model not found: {e}. Run train_model.py first.")
    MODEL_LOADED = False
    df = None
    cosine_sim = None
    indices = None


def get_recommendations(name, cosine_sim=cosine_sim):
    """Get top 10 restaurant recommendations based on content-based filtering."""
    try:
        # Find the index of the restaurant
        idx = indices[name]
        # Get pairwise similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get top 10 (excluding self)
        sim_scores = sim_scores[1:11]
        # Get restaurant indices
        rest_indices = [i[0] for i in sim_scores]
        # Return recommended restaurants
        result = df[['name', 'cuisines', 'mean_rating', 'cost']].iloc[rest_indices]
        result = result.rename(columns={'name': 'Restaurant', 'cuisines': 'cuisines', 
                                         'mean_rating': 'Mean Rating', 'cost': 'cost'})
        return result
    except KeyError:
        return None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/extractor', methods=['GET', 'POST'])
def extractor():
    return render_template('web.html', recommendation=None, restaurant_name='', error=None)


@app.route('/keywords', methods=['GET', 'POST'])
def keywords():
    if not MODEL_LOADED:
        return render_template('web.html', 
                               error="Model not loaded. Please run train_model.py first.",
                               recommendation=None, restaurant_name='')
    
    restaurant_name = request.args.get('restaurant_name', '') or request.form.get('restaurant_name', '')
    
    if not restaurant_name:
        return render_template('web.html', recommendation=None, restaurant_name='', error=None)
    
    recommendation = get_recommendations(restaurant_name, cosine_sim)
    
    if recommendation is None:
        return render_template('web.html', 
                               recommendation=None,
                               restaurant_name=restaurant_name,
                               error=f"Restaurant '{restaurant_name}' not found in database. Please check the name and try again.")
    
    # Convert to list of dicts for template
    rec_list = recommendation.to_dict('records')
    
    return render_template('result.html', 
                           recommendation=rec_list,
                           restaurant_name=restaurant_name)


@app.route('/search')
def search():
    """API endpoint to search restaurant names for autocomplete."""
    if not MODEL_LOADED:
        return {'restaurants': []}
    query = request.args.get('q', '').lower()
    if len(query) < 2:
        return {'restaurants': []}
    matches = [name for name in df['name'].unique() 
               if query in name.lower()][:10]
    return {'restaurants': matches}


if __name__ == '__main__':
    app.run(debug=True)
