# Restaurant Recommendation System

A full-stack AI-powered restaurant recommendation system using **Content-Based Filtering** 
with TF-IDF Vectorization and Cosine Similarity. Built with Flask, scikit-learn, and the 
Zomato Bangalore Restaurants dataset.

---

## 📁 Project Structure

```
restaurant_recommendation/
├── Dataset/
│   └── zomato.csv                          # Zomato Bangalore dataset (download from Kaggle)
├── Flask/
│   ├── app1.py                             # Main Flask application
│   ├── train_model.py                      # Model training script
│   ├── restaurant.pkl                      # Trained model (generated)
│   ├── restaurant1.csv                     # Processed data (generated)
│   ├── templates/
│   │   ├── index.html                      # Home page
│   │   ├── web.html                        # Recommendation input page
│   │   └── result.html                     # Results page
│   └── static/
│       ├── css/
│       │   └── main.css                    # All styles
│       ├── js/
│       │   └── main.js                     # Frontend JS
│       └── images/                         # Static images
├── Model/
│   └── Restaurant_Recommendation_System.ipynb   # Jupyter notebook with full analysis
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
Or install individually:
```bash
pip install pandas matplotlib seaborn plotly numpy scikit-learn Flask nltk
```

### Step 2: Download Dataset
Download the Zomato Bangalore dataset from Kaggle:
👉 https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants

Place `zomato.csv` in the `Dataset/` folder.

### Step 3: Train the Model
```bash
cd Flask
python train_model.py
```
This will:
- Load and clean the dataset
- Compute mean ratings
- Generate EDA visualizations
- Train the TF-IDF model
- Save `restaurant.pkl` and `restaurant1.csv`

### Step 4: Run the Flask App
```bash
cd Flask
python app1.py
```
Open your browser at: **http://127.0.0.1:5000**

---

## 🧠 How It Works

### Content-Based Filtering
1. **Data Collection**: Zomato Bangalore restaurant data with cuisines, ratings, votes, cost
2. **Pre-processing**: Clean ratings, handle missing values, standardize columns
3. **Feature Extraction**: TF-IDF Vectorization on cuisine text (bigrams)
4. **Similarity Computation**: Cosine Similarity between all restaurant vectors
5. **Recommendation**: Return top 10 most similar restaurants, sorted by rating

```
Restaurant A (Jalsa) → [North Indian, Chinese, Continental]
    ↓ TF-IDF
Vector A → cosine similarity → Vector B, C, D...
    ↓ Sort by similarity score
Top 10 recommended restaurants
```

### Algorithm
```python
# TF-IDF Vectorization
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(cuisines)

# Cosine Similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Get recommendations
sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
```

---

## 📊 Pages

| URL | Description |
|-----|-------------|
| `/` | Home page with project description |
| `/extractor` | Input page — enter restaurant name |
| `/keywords` | Results page — show top 10 recommendations |
| `/search?q=...` | Autocomplete API endpoint |

---

## 📦 Dataset

**Zomato Bangalore Restaurants**
- Source: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants
- ~51,000 restaurant records
- Features: name, cuisines, rate, votes, cost, location, rest_type

---

## 🛠 Technologies

| Technology | Purpose |
|-----------|---------|
| Python 3.9+ | Core language |
| Flask 2.x | Web framework |
| scikit-learn | TF-IDF + Cosine Similarity |
| pandas / numpy | Data processing |
| matplotlib / seaborn | Visualizations |
| plotly | Interactive plots |
| nltk | Text processing |
| pickle | Model serialization |
| HTML/CSS/JS | Frontend UI |

---

## 📖 Project Flow

1. **Data Collection** → Download Zomato Bangalore dataset from Kaggle
2. **Data Pre-processing** → Clean ratings, handle nulls, standardize columns  
3. **EDA** → Visualize cuisine distributions, rating patterns, cost analysis
4. **Content-Based Filtering** → TF-IDF + Cosine Similarity on cuisines
5. **Model Training** → Save trained model as `restaurant.pkl`
6. **Flask App** → Serve recommendations via web interface

---

## 🎯 Scenarios

- **Restaurant Visitors**: Discover restaurants similar to ones you love
- **Restaurant Owners**: Understand competitive landscape  
- **Food Delivery Platforms**: Power personalized discovery features

---

## 📝 Notes

- If the dataset is not found, a sample dataset with 30 Bangalore restaurants is auto-created for demo
- The model requires the dataset to be run once with `train_model.py` before the Flask app
- Restaurant names are case-sensitive in the current version
