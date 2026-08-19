# ====================================================================
# TRAIN_CODE: Her projenin (yaklasik/temsili) egitim kodu
# Not: Gercek model dosyalari burada yeniden egitilmiyor; bu bloklar
# arayuzdeki tahmin mantigiyla tutarli, projenin nasil egitildigini
# gosteren referans kodlardir.
# ====================================================================
TRAIN_CODE = {

"gold": '''import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

gold_path = kagglehub.dataset_download("altruistdelhite04/gold-price-data")
gold_df = pd.read_csv(os.path.join(gold_path, "gld_price_data.csv"))

X_gold = gold_df.drop(['Date', 'GLD'], axis=1)
y_gold = gold_df['GLD']

X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(X_gold, y_gold, test_size=0.2, random_state=42)

# 🔧 İYİLEŞTİRME: sabit parametre yerine RandomizedSearchCV (5-fold CV) ile hiper-parametre araması
param_grid_gold = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2]}
search_gold = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_grid_gold,
                                  n_iter=5, cv=5, random_state=42, n_jobs=-1)
search_gold.fit(X_tr_g, y_tr_g)
gold_model = search_gold.best_estimator_
joblib.dump(gold_model, 'models/gold_model.pkl')

y_pred_gold = gold_model.predict(X_te_g)
# best_params_ / best_score_ gerçek CV sonucunu ve seçilen parametreleri verir
print(f"En iyi parametreler: {search_gold.best_params_}  (5-fold CV R²={search_gold.best_score_:.4f})")

# Özellik önemleri (hangi gösterge altın fiyatını en çok etkiliyor?)
importances = pd.Series(gold_model.feature_importances_, index=X_gold.columns).sort_values(ascending=False)
print(importances.head(5))''',

"student": '''import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

student_path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
student_df = pd.read_csv(os.path.join(student_path, "StudentsPerformance.csv"))

student_df = pd.get_dummies(student_df, columns=[
    'gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'
], drop_first=True)

X_student = student_df.drop(['math score'], axis=1)
y_student = student_df['math score']

X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_student, y_student, test_size=0.2, random_state=42)

param_grid_student = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2]}
search_student = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_grid_student,
                                     n_iter=5, cv=5, random_state=42, n_jobs=-1)
search_student.fit(X_tr_s, y_tr_s)
student_model = search_student.best_estimator_
joblib.dump(student_model, 'models/student_model.pkl')

y_pred_student = student_model.predict(X_te_s)
print(f"En iyi parametreler: {search_student.best_params_}  (5-fold CV R²={search_student.best_score_:.4f})")

importances = pd.Series(student_model.feature_importances_, index=X_student.columns).sort_values(ascending=False)
print(importances.head(5))''',

"uber": '''import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

uber_path = kagglehub.dataset_download("yasserh/uber-fares-dataset")
uber_df = pd.read_csv(os.path.join(uber_path, "uber.csv"))

uber_df = uber_df.dropna()
uber_df = uber_df[(uber_df['pickup_longitude'] != 0) & (uber_df['fare_amount'] > 0)]

X_uber = uber_df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
y_uber = uber_df['fare_amount']

# Hız için ilk 50.000 satır
X_tr_u, X_te_u, y_tr_u, y_te_u = train_test_split(X_uber[:50000], y_uber[:50000], test_size=0.2, random_state=42)

param_grid_uber = {'n_estimators': [50, 100], 'max_depth': [None, 10, 15], 'min_samples_leaf': [1, 2]}
search_uber = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), param_grid_uber,
                                  n_iter=4, cv=3, random_state=42, n_jobs=-1)
search_uber.fit(X_tr_u, y_tr_u)
uber_model = search_uber.best_estimator_
joblib.dump(uber_model, 'models/uber_model.pkl')

y_pred_uber = uber_model.predict(X_te_u)
print(f"En iyi parametreler: {search_uber.best_params_}  (3-fold CV R²={search_uber.best_score_:.4f})")

importances = pd.Series(uber_model.feature_importances_, index=X_uber.columns).sort_values(ascending=False)
print(importances.head(5))''',

"mobile": '''import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

mobile_path = kagglehub.dataset_download("iabhishekofficial/mobile-price-classification")
mobile_df = pd.read_csv(os.path.join(mobile_path, "train.csv"))

features_mobile = ['battery_power', 'clock_speed', 'dual_sim', 'int_memory', 'm_dep',
                    'mobile_wt', 'n_cores', 'pc', 'ram', 'touch_screen', 'wifi']
X_mobile = mobile_df[features_mobile]
y_mobile = mobile_df['price_range']

X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_mobile, y_mobile, test_size=0.2, random_state=42)

param_grid_mobile = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2]}
search_mobile = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid_mobile,
                                    n_iter=5, cv=5, random_state=42, n_jobs=-1)
search_mobile.fit(X_tr_m, y_tr_m)
mobile_model = search_mobile.best_estimator_
joblib.dump(mobile_model, 'models/mobile_model.pkl')

y_pred_mobile = mobile_model.predict(X_te_m)
print(f"En iyi parametreler: {search_mobile.best_params_}  (5-fold CV Accuracy={search_mobile.best_score_:.4f})")

importances = pd.Series(mobile_model.feature_importances_, index=features_mobile).sort_values(ascending=False)
print(importances.head(5))

print(confusion_matrix(y_te_m, y_pred_mobile))
print(classification_report(y_te_m, y_pred_mobile))''',

"wine": '''import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

wine_path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
wine_df = pd.read_csv(os.path.join(wine_path, "winequality-red.csv"))

X_wine = wine_df.drop(['quality'], axis=1)
y_wine = wine_df['quality']

X_tr_w, X_te_w, y_tr_w, y_te_w = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

param_grid_wine = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2]}
search_wine = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid_wine,
                                  n_iter=5, cv=5, random_state=42, n_jobs=-1)
search_wine.fit(X_tr_w, y_tr_w)
wine_model = search_wine.best_estimator_
joblib.dump(wine_model, 'models/wine_model.pkl')

y_pred_wine = wine_model.predict(X_te_w)
print(f"En iyi parametreler: {search_wine.best_params_}  (5-fold CV Accuracy={search_wine.best_score_:.4f})")

importances = pd.Series(wine_model.feature_importances_, index=X_wine.columns).sort_values(ascending=False)
print(importances.head(5))

print(confusion_matrix(y_te_w, y_pred_wine))
print(classification_report(y_te_w, y_pred_wine))''',

"churn": '''import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

churn_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
churn_df = pd.read_csv(os.path.join(churn_path, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))

churn_df['TotalCharges'] = pd.to_numeric(churn_df['TotalCharges'], errors='coerce')
churn_df = churn_df.dropna()

churn_df['gender'] = churn_df['gender'].map({'Female': 0, 'Male': 1})
churn_df['Partner'] = churn_df['Partner'].map({'No': 0, 'Yes': 1})
churn_df['Dependents'] = churn_df['Dependents'].map({'No': 0, 'Yes': 1})
churn_df['PhoneService'] = churn_df['PhoneService'].map({'No': 0, 'Yes': 1})
churn_df['Churn'] = churn_df['Churn'].map({'No': 0, 'Yes': 1})

features_churn = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                   'PhoneService', 'MonthlyCharges', 'TotalCharges']
X_churn = churn_df[features_churn]
y_churn = churn_df['Churn']

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)

# Not: Churn veri seti dengesiz (Yes ~%27) -> class_weight='balanced' eklendi
param_grid_churn = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2]}
search_churn = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid_churn,
    n_iter=5, cv=5, random_state=42, n_jobs=-1)
search_churn.fit(X_tr_c, y_tr_c)
churn_model = search_churn.best_estimator_
joblib.dump(churn_model, 'models/churn_model.pkl')

y_pred_churn = churn_model.predict(X_te_c)
print(f"En iyi parametreler: {search_churn.best_params_}  (5-fold CV Accuracy={search_churn.best_score_:.4f})")

importances = pd.Series(churn_model.feature_importances_, index=features_churn).sort_values(ascending=False)
print(importances.head(5))

print(confusion_matrix(y_te_c, y_pred_churn))
print(classification_report(y_te_c, y_pred_churn, target_names=["Kalan", "Terk Eden"]))''',

"nba": '''import os
import pandas as pd
import kagglehub
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

nba_path = kagglehub.dataset_download("justinas/nba-players-data")
nba_df = pd.read_csv(os.path.join(nba_path, "all_seasons.csv"))

features_nba = ['pts', 'reb', 'ast']
X_nba = nba_df[features_nba].dropna()

scaler_nba = StandardScaler()
X_nba_scaled = scaler_nba.fit_transform(X_nba)

# 🔧 İYİLEŞTİRME: sabit n_clusters=4 yerine silhouette skoruna göre otomatik k seçimi
best_k, best_score, kmeans_nba = None, -1, None
for k in [3, 4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_nba_scaled)
    score = silhouette_score(X_nba_scaled, km.labels_)
    print(f"k={k}: Silhouette={score:.4f}")
    if score > best_score:
        best_k, best_score, kmeans_nba = k, score, km

joblib.dump(kmeans_nba, 'models/nba_model.pkl')
joblib.dump(scaler_nba, 'models/nba_scaler.pkl')
print(f"Seçilen en iyi k={best_k}  Silhouette={best_score:.4f}")''',

"creditcard": '''import os
import pandas as pd
import kagglehub
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

cc_path = kagglehub.dataset_download("arjunbhasin2013/ccdata")
cc_df = pd.read_csv(os.path.join(cc_path, "CC GENERAL.csv"))

cc_df['CREDIT_LIMIT'] = cc_df['CREDIT_LIMIT'].fillna(cc_df['CREDIT_LIMIT'].median())
cc_df['MINIMUM_PAYMENTS'] = cc_df['MINIMUM_PAYMENTS'].fillna(cc_df['MINIMUM_PAYMENTS'].median())

features_cc = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']
X_cc = cc_df[features_cc]

scaler_cc = StandardScaler()
X_cc_scaled = scaler_cc.fit_transform(X_cc)

best_k, best_score, kmeans_cc = None, -1, None
for k in [3, 4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cc_scaled)
    score = silhouette_score(X_cc_scaled, km.labels_)
    print(f"k={k}: Silhouette={score:.4f}")
    if score > best_score:
        best_k, best_score, kmeans_cc = k, score, km

joblib.dump(kmeans_cc, 'models/cc_model.pkl')
joblib.dump(scaler_cc, 'models/cc_scaler.pkl')
print(f"Seçilen en iyi k={best_k}  Silhouette={best_score:.4f}")''',

"spotify": '''import os
import pandas as pd
import kagglehub
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

spot_path = kagglehub.dataset_download("yashdev01/spotify-tracks-dataset")
csv_files = [f for f in os.listdir(spot_path) if f.endswith('.csv')]
csv_file_path = os.path.join(spot_path, csv_files[0])
spot_df = pd.read_csv(csv_file_path)

features_spot = ['danceability', 'energy', 'loudness']
X_spot = spot_df[features_spot].dropna()

scaler_spot = StandardScaler()
X_spot_scaled = scaler_spot.fit_transform(X_spot)

best_k, best_score, kmeans_spot = None, -1, None
for k in [3, 4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_spot_scaled)
    score = silhouette_score(X_spot_scaled, km.labels_)
    print(f"k={k}: Silhouette={score:.4f}")
    if score > best_score:
        best_k, best_score, kmeans_spot = k, score, km

joblib.dump(kmeans_spot, 'models/spotify_model.pkl')
joblib.dump(scaler_spot, 'models/spotify_scaler.pkl')
print(f"Seçilen en iyi k={best_k}  Silhouette={best_score:.4f}")''',

"drowsy": '''# ⚠️ Bu modül şu an GERÇEK bir model kullanmıyor, rastgele simülasyon yapıyor.
# train.py içinde bu modül için hiçbir eğitim kodu yok.
# Gerçek bir "sürücü uyuklama tespiti" tipik olarak MediaPipe FaceMesh ile
# EAR (Eye Aspect Ratio) hesaplayıp bir eşik değeriyle karar verir.

EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 15''',

"mask": '''import os
import cv2
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

mask_path = kagglehub.dataset_download("omkargurav/face-mask-dataset")

data, labels = [], []
for label, folder in enumerate(['with_mask', 'without_mask']):
    folder_path = os.path.join(mask_path, 'data', folder)
    for img_name in os.listdir(folder_path)[:200]:
        img = cv2.imread(os.path.join(folder_path, img_name))
        img = cv2.resize(img, (64, 64))
        data.append(img.flatten())
        labels.append(label)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid_mask = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
search_mask = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid_mask,
                                  n_iter=4, cv=3, random_state=42, n_jobs=-1)
search_mask.fit(X_train, y_train)
mask_model = search_mask.best_estimator_
joblib.dump(mask_model, 'models/mask_model.pkl')

y_pred_mask = mask_model.predict(X_test)
print(f"En iyi parametreler: {search_mask.best_params_}  (3-fold CV Accuracy={search_mask.best_score_:.4f})")
print(confusion_matrix(y_test, y_pred_mask))
print(classification_report(y_test, y_pred_mask, target_names=["Maskeli", "Maskesiz"]))''',

"hand": '''# Bu modül eğitim GEREKTİRMEZ.
# Google'ın önceden eğitilmiş (pretrained) MediaPipe Hands modelini kullanır.

import mediapipe as mp

mp_hands = mp.solutions.hands
with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    results = hands.process(rgb_image)''',

"sms": '''import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib

spam_path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
spam_df = pd.read_csv(os.path.join(spam_path, "spam.csv"), encoding='latin-1')

# 🔧 DÜZELTME: Ham CSV'de v1/v2 dışında neredeyse tamamen boş
# "Unnamed: 2/3/4" sütunları var. Sadece gerekli sütunları seçiyoruz
# (aksi halde eksik-değer temizliği veri setinin neredeyse tamamını silerdi).
spam_df = spam_df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
spam_df['label'] = spam_df['label'].map({'ham': 0, 'spam': 1})

tfidf_spam = TfidfVectorizer(stop_words='english', max_features=3000)
X_spam = tfidf_spam.fit_transform(spam_df['text'])
y_spam = spam_df['label']

X_tr_sp, X_te_sp, y_tr_sp, y_te_sp = train_test_split(X_spam, y_spam, test_size=0.2, random_state=42)
spam_model = MultinomialNB()
spam_model.fit(X_tr_sp, y_tr_sp)

joblib.dump(spam_model, 'models/spam_model.pkl')
joblib.dump(tfidf_spam, 'models/spam_vectorizer.pkl')

y_pred_spam = spam_model.predict(X_te_sp)
print(confusion_matrix(y_te_sp, y_pred_spam))
print(classification_report(y_te_sp, y_pred_spam, target_names=["Ham (Normal)", "Spam"]))''',

"imdb_sentiment": '''import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

imdb_path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
imdb_df = pd.read_csv(os.path.join(imdb_path, "IMDB Dataset.csv"))

# 🔧 DÜZELTME: satır silmek yerine EKSİK DEĞER İÇEREN SÜTUNLARI tamamen çıkarıyoruz
eksik_sutunlar = imdb_df.columns[imdb_df.isnull().any()].tolist()
if eksik_sutunlar:
    imdb_df = imdb_df.drop(columns=eksik_sutunlar)

imdb_df['sentiment'] = imdb_df['sentiment'].map({'negative': 0, 'positive': 1})

# Hız için 10.000 satır
imdb_sub = imdb_df.sample(10000, random_state=42)
tfidf_imdb = TfidfVectorizer(stop_words='english', max_features=5000)
X_imdb = tfidf_imdb.fit_transform(imdb_sub['review'])
y_imdb = imdb_sub['sentiment']

X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(X_imdb, y_imdb, test_size=0.2, random_state=42)
imdb_model = LogisticRegression(max_iter=200)
imdb_model.fit(X_tr_i, y_tr_i)

joblib.dump(imdb_model, 'models/imdb_model.pkl')
joblib.dump(tfidf_imdb, 'models/imdb_vectorizer.pkl')

y_pred_imdb = imdb_model.predict(X_te_i)
print(confusion_matrix(y_te_i, y_pred_imdb))
print(classification_report(y_te_i, y_pred_imdb, target_names=["Negatif", "Pozitif"]))''',

"fake_news": '''import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

news_path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

fake_df = pd.read_csv(os.path.join(news_path, "Fake.csv"))
real_df = pd.read_csv(os.path.join(news_path, "True.csv"))
fake_df['label'] = 1  # Sahte
real_df['label'] = 0  # Gerçek

news_df = pd.concat([
    fake_df.sample(4000, random_state=42), real_df.sample(4000, random_state=42)
]).reset_index(drop=True)

tfidf_news = TfidfVectorizer(stop_words='english', max_features=5000)
X_news = tfidf_news.fit_transform(news_df['text'])
y_news = news_df['label']

X_tr_n, X_te_n, y_tr_n, y_te_n = train_test_split(X_news, y_news, test_size=0.2, random_state=42)
news_model = LogisticRegression()
news_model.fit(X_tr_n, y_tr_n)

joblib.dump(news_model, 'models/news_model.pkl')
joblib.dump(tfidf_news, 'models/news_vectorizer.pkl')

y_pred_news = news_model.predict(X_te_n)
print(confusion_matrix(y_te_n, y_pred_news))
print(classification_report(y_te_n, y_pred_news, target_names=["Gerçek", "Sahte"]))''',

"movie_rec": '''import os
import pandas as pd
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

movie_path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
movies = pd.read_csv(os.path.join(movie_path, "tmdb_5000_movies.csv"))

movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['features'] = movies['overview'] + " " + movies['genres']

# En popüler 1500 film (hafıza optimizasyonu)
movies_sub = movies.sort_values('popularity', ascending=False).head(1500).reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_sub['features'])
movie_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

joblib.dump(movies_sub[['title']], 'models/movie_data.pkl')
joblib.dump(movie_sim, 'models/movie_similarity.pkl')
# Not: içerik tabanlı (unsupervised) öneri — accuracy/R² gibi tek metrik yok''',

"book_rec": '''import os
import pandas as pd
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

book_path = kagglehub.dataset_download("jealousleopard/goodreadsbooks")
books = pd.read_csv(os.path.join(book_path, "books.csv"), on_bad_lines='skip')

books['features'] = books['title'].fillna('') + " " + books['authors'].fillna('')
books_sub = books.head(1500).reset_index(drop=True)

tfidf_book = TfidfVectorizer(stop_words='english')
tfidf_book_matrix = tfidf_book.fit_transform(books_sub['features'])
book_sim = cosine_similarity(tfidf_book_matrix, tfidf_book_matrix)

joblib.dump(books_sub[['title', 'authors']], 'models/book_data.pkl')
joblib.dump(book_sim, 'models/book_similarity.pkl')
# Not: içerik tabanlı (unsupervised) öneri — tek bir başarı metriği yok''',

"song_rec": '''import os
import pandas as pd
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

song_path = kagglehub.dataset_download("yashdev01/spotify-tracks-dataset")
csv_files = [f for f in os.listdir(song_path) if f.endswith('.csv')]
csv_file_path = os.path.join(song_path, csv_files[0])
songs = pd.read_csv(csv_file_path)

songs['features'] = songs['track_name'].fillna('') + " " + songs['artists'].fillna('') + " " + songs['track_genre'].fillna('')
songs_clean = songs.drop_duplicates(subset=['track_name']).head(1500).reset_index(drop=True)

tfidf_song = TfidfVectorizer(stop_words='english')
tfidf_song_matrix = tfidf_song.fit_transform(songs_clean['features'])
song_sim = cosine_similarity(tfidf_song_matrix, tfidf_song_matrix)

joblib.dump(songs_clean[['track_name', 'artists']], 'models/song_data.pkl')
joblib.dump(song_sim, 'models/song_similarity.pkl')
# Not: içerik tabanlı (unsupervised) öneri — tek bir başarı metriği yok''',

"stock": '''import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

ticker = yf.Ticker("AAPL")
stock_df = ticker.history(period="5y")
stock_df.columns = [col.lower() for col in stock_df.columns]

stock_df['close_lag1'] = stock_df['close'].shift(1)
stock_df = stock_df.dropna()

X_stock = stock_df[['close_lag1']]
y_stock = stock_df['close']

# Zaman serisi -> kronolojik %80/%20 böl (test için)
split_idx = int(len(X_stock) * 0.8)
X_tr, X_te = X_stock.iloc[:split_idx], X_stock.iloc[split_idx:]
y_tr, y_te = y_stock.iloc[:split_idx], y_stock.iloc[split_idx:]

eval_model = LinearRegression().fit(X_tr, y_tr)
y_pred = eval_model.predict(X_te)
r2 = r2_score(y_te, y_pred)
rmse = mean_squared_error(y_te, y_pred) ** 0.5
mae = mean_absolute_error(y_te, y_pred)

# Üretim modeli TÜM veriyle nihai olarak eğitilir
stock_model = LinearRegression()
stock_model.fit(X_stock, y_stock)
joblib.dump(stock_model, 'models/stock_model.pkl')
joblib.dump(stock_df['close'].tail(30).values, 'models/stock_recent.pkl')''',

"weather": '''import os
import pandas as pd
import kagglehub
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

weather_path = kagglehub.dataset_download("sumanthvrao/daily-climate-time-series-data")
weather_df = pd.read_csv(os.path.join(weather_path, "DailyDelhiClimateTrain.csv"))

weather_df['meantemp_Lag1'] = weather_df['meantemp'].shift(1)
weather_df = weather_df.dropna()

X_weather = weather_df[['meantemp_Lag1']]
y_weather = weather_df['meantemp']

split_idx = int(len(X_weather) * 0.8)
X_tr, X_te = X_weather.iloc[:split_idx], X_weather.iloc[split_idx:]
y_tr, y_te = y_weather.iloc[:split_idx], y_weather.iloc[split_idx:]

eval_model = LinearRegression().fit(X_tr, y_tr)
y_pred = eval_model.predict(X_te)
r2 = r2_score(y_te, y_pred)
rmse = mean_squared_error(y_te, y_pred) ** 0.5
mae = mean_absolute_error(y_te, y_pred)

weather_model = LinearRegression()
weather_model.fit(X_weather, y_weather)
joblib.dump(weather_model, 'models/weather_model.pkl')
joblib.dump(weather_df['meantemp'].tail(30).values, 'models/weather_recent.pkl')''',

"walmart": '''import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Kaggle kısıtlamasını aşmak için Walmart trendlerine uygun sentetik zaman serisi
np.random.seed(42)
dates = pd.date_range(start="2010-02-05", periods=143, freq="W")
weekly_sales = 1500000 + np.sin(np.linspace(0, 20, 143)) * 300000 + np.random.normal(0, 50000, 143)

walmart_sub = pd.DataFrame({'Date': dates, 'Weekly_Sales': weekly_sales}).sort_values('Date').reset_index(drop=True)
walmart_sub['Sales_Lag1'] = walmart_sub['Weekly_Sales'].shift(1)
walmart_sub = walmart_sub.dropna()

X_walmart = walmart_sub[['Sales_Lag1']]
y_walmart = walmart_sub['Weekly_Sales']

split_idx = int(len(X_walmart) * 0.8)
X_tr, X_te = X_walmart.iloc[:split_idx], X_walmart.iloc[split_idx:]
y_tr, y_te = y_walmart.iloc[:split_idx], y_walmart.iloc[split_idx:]

eval_model = LinearRegression().fit(X_tr, y_tr)
y_pred = eval_model.predict(X_te)
r2 = r2_score(y_te, y_pred)
rmse = mean_squared_error(y_te, y_pred) ** 0.5
mae = mean_absolute_error(y_te, y_pred)

walmart_model = LinearRegression()
walmart_model.fit(X_walmart, y_walmart)
joblib.dump(walmart_model, 'models/walmart_model.pkl')
joblib.dump(walmart_sub['Weekly_Sales'].tail(15).values, 'models/walmart_recent.pkl')''',

"social_media_viz": '''# Bu sekmede bir ML modeli EĞİTİLMİYOR — sentetik veri üretilip
# doğrudan Streamlit'te görselleştiriliyor (Kaggle bağımlılığı yok).
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 500
platforms = ['Instagram', 'TikTok', 'X (Twitter)', 'YouTube']
genders = ['Male', 'Female', 'Non-binary']
emotions = ['Happy', 'Anxious', 'Sad', 'Neutral', 'Bored']

sm_data = {
    'Age': np.random.randint(13, 65, n_samples),
    'Gender': np.random.choice(genders, n_samples),
    'Platform': np.random.choice(platforms, n_samples),
    'Daily_Usage_Time (minutes)': np.random.randint(15, 360, n_samples),
    'Dominant_Emotion': np.random.choice(emotions, n_samples),
}
pd.DataFrame(sm_data).to_csv('models/social_media_viz.csv', index=False)''',

"co2_viz": '''# Bu sekmede de bir model yok; ülke bazlı gerçekçi CO2 trend simülasyonu üretiliyor.
import numpy as np
import pandas as pd

years = list(range(1990, 2024))
countries = ['United States', 'China', 'United Kingdom', 'Germany', 'India', 'Japan']
co2_records = []
for country in countries:
    base_total = np.random.randint(500, 3000)
    for year in years:
        idx = year - 1990
        trend = (1.0 + idx * 0.04) if country in ['China', 'India'] else (1.0 - idx * 0.01)
        total = max(100, base_total * trend)
        co2_records.append({'Country': country, 'Year': year, 'Total': round(total, 2)})

pd.DataFrame(co2_records).to_csv('models/co2_emissions_viz.csv', index=False)''',

"ecommerce_viz": '''# Bu sekmede de model yok; sentetik e-ticaret işlem verisi üretiliyor.
import numpy as np
import pandas as pd

products = ['Wireless Mouse', 'Mechanical Keyboard', 'Gaming Monitor', 'USB-C Cable',
            'Bluetooth Headphones', 'Laptop Stand', 'Desk Mat', 'Smartphone Case',
            'Webcam 1080p', 'LED Desk Lamp']
ec_countries = ['United Kingdom', 'Germany', 'France', 'Spain', 'Italy', 'Netherlands']

ec_records = []
for i in range(1000):
    qty = np.random.randint(1, 5)
    unit_price = round(float(np.random.uniform(10.0, 150.0)), 2)
    ec_records.append({
        'InvoiceNo': str(10000 + i),
        'Description': np.random.choice(products),
        'Quantity': qty, 'UnitPrice': unit_price,
        'Country': np.random.choice(ec_countries),
        'Total_Price': round(qty * unit_price, 2),
    })
pd.DataFrame(ec_records).to_csv('models/ecommerce_viz.csv', index=False)''',

"pneumonia": '''import os
import cv2
import numpy as np
import kagglehub
from sklearn.utils import shuffle as sk_shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

pneu_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

data_p, labels_p = [], []
base_dir = os.path.join(pneu_path, "chest_xray", "train")
for label, subfolder in enumerate(["NORMAL", "PNEUMONIA"]):
    folder_path = os.path.join(base_dir, subfolder)
    for img_name in os.listdir(folder_path)[:150]:
        img = cv2.imread(os.path.join(folder_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64)) / 255.0
        data_p.append(img)
        labels_p.append(label)

X_p = np.array(data_p).reshape(-1, 64, 64, 1)
y_p = np.array(labels_p)

# 🚨 KRİTİK DÜZELTME: veri [NORMAL...NORMAL, PNEUMONIA...PNEUMONIA] sıralı geliyor.
# Keras'ın validation_split'i veriyi KARIŞTIRMADAN sondan %20'sini ayırır — bu da
# validation setinin TEK bir sınıftan oluşmasına ve val_accuracy=0.0 çıkmasına
# sebep oluyordu. Çözüm: fit'ten önce X/y'yi birlikte karıştırmak.
X_p, y_p = sk_shuffle(X_p, y_p, random_state=42)

model_p = Sequential([
    Input(shape=(64, 64, 1)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model_p.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_p = model_p.fit(X_p, y_p, epochs=3, batch_size=32, validation_split=0.2, verbose=0)

# .h5 yerine güncel .keras formatı kullanılıyor
model_p.save('models/pneumonia_model.keras')

accuracy = history_p.history['accuracy'][-1]
val_accuracy = history_p.history['val_accuracy'][-1]
loss = history_p.history['loss'][-1]''',

"face_emotion": '''import os
import cv2
import numpy as np
import kagglehub
from sklearn.utils import shuffle as sk_shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

fer_path = kagglehub.dataset_download("msambare/fer2013")

emotions = ["angry", "happy", "sad"]
data_f, labels_f = [], []
train_dir = os.path.join(fer_path, "train")
for label, emotion in enumerate(emotions):
    folder_path = os.path.join(train_dir, emotion)
    for img_name in os.listdir(folder_path)[:150]:
        img = cv2.imread(os.path.join(folder_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48)) / 255.0
        data_f.append(img)
        labels_f.append(label)

X_f = np.array(data_f).reshape(-1, 48, 48, 1)
y_f = np.array(labels_f)

# 🚨 KRİTİK DÜZELTME: aynı shuffle sorunu burada da var (angry/happy/sad sıralı)
X_f, y_f = sk_shuffle(X_f, y_f, random_state=42)

model_f = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax'),
])
model_f.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_f = model_f.fit(X_f, y_f, epochs=3, batch_size=32, validation_split=0.2, verbose=0)
model_f.save('models/fer_model.keras')

accuracy = history_f.history['accuracy'][-1]
val_accuracy = history_f.history['val_accuracy'][-1]
loss = history_f.history['loss'][-1]''',

"text_gen": '''import joblib

# 🔧 İYİLEŞTİRME: daha zengin bir Markov Zinciri için genişletilmiş corpus (Hamlet)
shakespeare_text = (
    "to be or not to be that is the question whether tis nobler in the mind "
    "to suffer the slings and arrows of outrageous fortune or to take arms "
    "against a sea of troubles and by opposing end them to die to sleep "
    "no more and by a sleep to say we end the heart ache and the thousand "
    "natural shocks that flesh is heir to tis a consummation devoutly to be wished "
    "to die to sleep to sleep perchance to dream ay theres the rub "
    "for in that sleep of death what dreams may come when we have shuffled off this mortal coil "
    "must give us pause there is the respect that makes calamity of so long life"
)
words = shakespeare_text.split()
markov_chain = {}

for i in range(len(words) - 1):
    current_word, next_word = words[i], words[i + 1]
    markov_chain.setdefault(current_word, []).append(next_word)

joblib.dump(markov_chain, 'models/text_robot_model.pkl')''',

"farm_agent": '''# Bu ajan bir ML modeli DEĞİL, kural tabanlı (rule-based) bir karar mekanizmasıdır.
# train.py içinde bu modül için bir eğitim adımı yoktur.

def sulama_karari(soil_moisture, temperature, sunlight):
    if soil_moisture < 30 and temperature > 30 and sunlight == "Yüksek":
        return "Acil Yoğun Sulama", 45
    elif soil_moisture < 40 and temperature > 15:
        return "Standart Sulama", 20
    else:
        return "Sulama Gerekli Değil", 0''',

"faq_agent": '''# Kural tabanlı (rule-based) niyet analizi — anahtar kelime eşleştirmesi.
# train.py içinde bu modül için bir eğitim adımı yoktur.

def niyet_analizi(mesaj):
    mesaj = mesaj.lower()
    if any(k in mesaj for k in ["iade", "iptal", "para"]):
        return "Finans / İade ve İptal Talebi"
    elif any(k in mesaj for k in ["kargo", "sipariş", "nerede"]):
        return "Lojistik / Kargo ve Teslimat Takibi"
    else:
        return "Genel / Teşekkür - Bilgi Talebi"''',

"autonomous_car": '''# Sensör eşiklerine dayalı kural tabanlı bir karar mekanizması.
# train.py içinde bu modül için bir eğitim adımı yoktur.

def surus_karari(sensor_distance, lane_status, parking_slot):
    seritte_kal = lane_status == "Net Görünür"
    acil_fren = sensor_distance < 20
    park_et = parking_slot and sensor_distance > 30
    return seritte_kal, acil_fren, park_et''',

}
