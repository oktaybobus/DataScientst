# DataScientst -- 30-Project Machine Learning & AI Portfolio

A comprehensive data science portfolio covering **10 categories** with **30 projects** and **34 trained models**, served through a single interactive Streamlit web application.

All models are trained on real-world **Kaggle datasets** using scikit-learn, TensorFlow/Keras, and classical ML techniques. Models are hosted on [Hugging Face Hub](https://huggingface.co/OKTAYBBS/DataScientst-models) and automatically downloaded at runtime.

## Live Demo

| Platform | Link |
|----------|------|
| Streamlit App | [oktaybobus-datascientst.streamlit.app](https://oktaybobus-datascientst.streamlit.app) |
| HF Portfolio | [huggingface.co/spaces/OKTAYBBS/DataScientst](https://huggingface.co/spaces/OKTAYBBS/DataScientst) |
| HF Models | [huggingface.co/OKTAYBBS/DataScientst-models](https://huggingface.co/OKTAYBBS/DataScientst-models) |

## Projects Overview

### 1. Regression (3 projects)
| Project | Model | Metric |
|---------|-------|--------|
| Gold Price Prediction | RandomForestRegressor | R² = 0.990 |
| Student Exam Score Prediction | RandomForestRegressor | R² = 0.849 |
| Uber/Taxi Fare Prediction | RandomForestRegressor | R² = 0.778 |

### 2. Classification (3 projects)
| Project | Model | Metric |
|---------|-------|--------|
| Mobile Device Price Segment | RandomForestClassifier | Accuracy = 81.2% |
| Wine Quality Classification | RandomForestClassifier | Accuracy = 67.5% |
| Customer Churn Prediction | RandomForestClassifier | Accuracy = 78.9% |

### 3. Clustering (3 projects)
| Project | Model | Metric |
|---------|-------|--------|
| NBA Player Performance Clustering | KMeans | Silhouette = 0.452 |
| Credit Card Customer Segmentation | KMeans | Silhouette = 0.531 |
| Spotify Song Style Clustering | KMeans | Silhouette = 0.327 |

### 4. Computer Vision (3 projects)
| Project | Model | Metric |
|---------|-------|--------|
| Driver Drowsiness Detection | MediaPipe + EAR | -- |
| Face Mask Detection | RandomForestClassifier | Accuracy = 82.5% |
| Hand Gesture & Finger Counting | MediaPipe Hands | -- |

### 5. Natural Language Processing (3 projects)
| Project | Model | Metric |
|---------|-------|--------|
| SMS Spam Detection | MultinomialNB + TF-IDF | Accuracy = 98.0% |
| IMDb Sentiment Analysis | LogisticRegression + TF-IDF | Accuracy = 87.3% |
| Fake News Detection | LogisticRegression + TF-IDF | Accuracy = 97.6% |

### 6. Recommendation Systems (3 projects)
| Project | Method |
|---------|--------|
| Movie Recommendation System | TF-IDF + Cosine Similarity |
| Book Recommendation Engine | TF-IDF + Cosine Similarity |
| Music Recommendation System | TF-IDF + Cosine Similarity |

### 7. Time Series (3 projects)
| Project | Model | Metric |
|---------|-------|--------|
| Stock Price Prediction (AAPL) | LinearRegression | R² = 0.975 |
| Weather Temperature Prediction | LinearRegression | R² = 0.912 |
| Store Sales Prediction | LinearRegression | R² = 0.767 |

### 8. Data Visualization (3 projects)
| Project | Tools |
|---------|-------|
| Global Social Media Statistics | Plotly Express |
| Climate Change & CO2 Emissions | Plotly Express |
| E-Commerce Sales Dashboard | Plotly Express |

### 9. Deep Learning (3 projects)
| Project | Model | Metric |
|---------|-------|--------|
| Pneumonia Detection from X-Ray | CNN (Keras) | Val Acc = 92.5% |
| Facial Emotion Recognition | CNN (Keras) | Val Acc = 65.4% |
| Text Generation Bot | Markov Chain | -- |

### 10. AI Agents (3 projects)
| Project | Type |
|---------|------|
| Smart Farming & Irrigation Agent | Rule-based Decision Tree |
| FAQ Support & Intent Analysis Agent | Intent Classification |
| Autonomous Vehicle Simulation Agent | Sensor-based Simulation |

## Architecture

```
app.py                  # Main Streamlit router (55 lines)
model_loader.py         # HF Hub model downloader with local fallback
components.py           # Shared UI components (metrics, dataset info)
dataset_info.py         # Dataset descriptions and metadata
train_codes.py          # Training code snippets for display
categories/
  regression.py         # 3 regression projects
  classification.py     # 3 classification projects
  clustering.py         # 3 clustering projects
  computer_vision.py    # 3 CV projects
  nlp.py                # 3 NLP projects
  recommendation.py     # 3 recommendation projects
  time_series.py        # 3 time series projects
  data_viz.py           # 3 visualization projects
  deep_learning.py      # 3 deep learning projects
  ai_agents.py          # 3 AI agent projects
```

## Local Setup

```bash
git clone https://github.com/oktaybobus/DataScientst.git
cd DataScientst
pip install -r requirements.txt
streamlit run app.py
```

Models are automatically downloaded from [Hugging Face Hub](https://huggingface.co/OKTAYBBS/DataScientst-models) on first run and cached locally.

## Tech Stack

- **UI:** Streamlit, Plotly Express
- **ML:** scikit-learn, Joblib
- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV, MediaPipe
- **Data:** Pandas, NumPy, Kaggle Hub API
- **Model Hosting:** Hugging Face Hub
