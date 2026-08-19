import pandas as pd

# ====================================================================
# DATASET_INFO: Her projenin veri seti aciklamasi + ornek veri
# ====================================================================

DATASET_INFO = {

"gold": {
    "name": "Gold Price Dataset",
    "source": "Kaggle - altruistdelhite04/gold-price-data",
    "size": "~2290 satır, 2008–2018 arası günlük veri",
    "description": "S&P 500, ham petrol (USO), gümüş (SLV) ve EUR/USD paritesi kullanılarak "
                   "GLD (altın ETF) fiyatının tahmin edildiği regresyon veri seti.",
    "columns": {
        "Date": "İşlem tarihi",
        "SPX": "S&P 500 endeks değeri",
        "GLD": "Hedef değişken — Altın ETF fiyatı ($)",
        "USO": "Ham petrol fiyat endeksi",
        "SLV": "Gümüş fiyatı ($)",
        "EUR/USD": "Euro/Dolar paritesi",
    },
    "sample": pd.DataFrame({
        "Date": ["1/2/2008", "1/3/2008", "1/4/2008"],
        "SPX": [1447.16, 1447.16, 1411.63],
        "GLD": [84.86, 85.57, 85.13],
        "USO": [78.47, 78.37, 77.31],
        "SLV": [15.18, 15.29, 15.17],
        "EUR/USD": [1.4712, 1.4735, 1.4676],
    }),
},

"student": {
    "name": "Students Performance in Exams",
    "source": "Kaggle - spscientist/students-performance-in-exams",
    "size": "1000 öğrenci gözlemi",
    "description": "Öğrencilerin demografik/sosyal özellikleri ve okuma/yazma notlarına göre "
                    "matematik sınav puanının tahmin edildiği regresyon veri seti.",
    "columns": {
        "gender": "Cinsiyet (male/female)",
        "race/ethnicity": "Etnik grup (Group A–E)",
        "parental level of education": "Ebeveyn eğitim seviyesi",
        "lunch": "Öğle yemeği tipi (standard / free-reduced)",
        "test preparation course": "Hazırlık kursu (none / completed)",
        "math score": "Hedef değişken — Matematik notu (0–100)",
        "reading score": "Okuma notu (0–100)",
        "writing score": "Yazma notu (0–100)",
    },
    "sample": pd.DataFrame({
        "gender": ["female", "male", "female"],
        "race/ethnicity": ["group B", "group C", "group B"],
        "parental level of education": ["bachelor's degree", "some college", "master's degree"],
        "lunch": ["standard", "standard", "standard"],
        "test preparation course": ["none", "completed", "none"],
        "math score": [72, 69, 90],
        "reading score": [72, 90, 95],
        "writing score": [74, 88, 93],
    }),
},

"uber": {
    "name": "Uber Fares Dataset",
    "source": "Kaggle - yasserh/uber-fares-dataset",
    "size": "~200.000 yolculuk kaydı (New York)",
    "description": "Alış/varış koordinatları ve yolcu sayısına göre taksi/Uber ücretinin "
                    "tahmin edildiği regresyon veri seti.",
    "columns": {
        "pickup_longitude": "Alış boylamı",
        "pickup_latitude": "Alış enlemi",
        "dropoff_longitude": "Varış boylamı",
        "dropoff_latitude": "Varış enlemi",
        "passenger_count": "Yolcu sayısı",
        "fare_amount": "Hedef değişken — Yolculuk ücreti ($)",
    },
    "sample": pd.DataFrame({
        "pickup_longitude": [-73.9997, -73.9836, -74.0060],
        "pickup_latitude": [40.7217, 40.7284, 40.7128],
        "dropoff_longitude": [-73.9910, -73.9663, -73.9654],
        "dropoff_latitude": [40.7500, 40.7654, 40.7829],
        "passenger_count": [1, 2, 1],
        "fare_amount": [7.5, 12.0, 9.3],
    }),
},

"mobile": {
    "name": "Mobile Price Classification",
    "source": "Kaggle - iabhishekofficial/mobile-price-classification",
    "size": "2000 telefon kaydı",
    "description": "Telefon donanım özelliklerine göre fiyat segmentinin (0–3) tahmin edildiği "
                    "çok sınıflı sınıflandırma veri seti.",
    "columns": {
        "battery_power": "Batarya gücü (mAh)",
        "clock_speed": "İşlemci hızı (GHz)",
        "dual_sim": "Çift SIM desteği (0/1)",
        "int_memory": "Dahili hafıza (GB)",
        "m_dep": "Kalınlık (cm)",
        "mobile_wt": "Ağırlık (gram)",
        "n_cores": "Çekirdek sayısı",
        "pc": "Arka kamera çözünürlüğü (MP)",
        "ram": "RAM (MB)",
        "touch_screen": "Dokunmatik ekran (0/1)",
        "wifi": "Wi-Fi desteği (0/1)",
        "price_range": "Hedef değişken — 0: Ucuz, 1: Normal, 2: Pahalı, 3: Amiral gemisi",
    },
    "sample": pd.DataFrame({
        "battery_power": [842, 1021, 563],
        "clock_speed": [2.2, 0.5, 1.0],
        "dual_sim": [0, 1, 1],
        "int_memory": [7, 53, 41],
        "m_dep": [0.6, 0.7, 0.9],
        "mobile_wt": [188, 136, 145],
        "n_cores": [2, 3, 5],
        "pc": [2, 6, 6],
        "ram": [2549, 2631, 2603],
        "touch_screen": [0, 1, 1],
        "wifi": [1, 0, 0],
        "price_range": [1, 2, 2],
    }),
},

"wine": {
    "name": "Red Wine Quality",
    "source": "UCI Machine Learning Repository - Wine Quality (red)",
    "size": "1599 şarap örneği",
    "description": "Kırmızı şarabın fizikokimyasal özelliklerine göre kalite puanının (0–10) "
                    "tahmin edildiği sınıflandırma/regresyon veri seti.",
    "columns": {
        "fixed acidity": "Sabit asitlik",
        "volatile acidity": "Uçucu asitlik",
        "citric acid": "Sitrik asit",
        "residual sugar": "Kalan şeker",
        "chlorides": "Klorürler",
        "free sulfur dioxide": "Serbest kükürt dioksit",
        "total sulfur dioxide": "Toplam kükürt dioksit",
        "density": "Yoğunluk",
        "pH": "pH değeri",
        "sulphates": "Sülfatlar",
        "alcohol": "Alkol oranı (%)",
        "quality": "Hedef değişken — Kalite puanı (0–10)",
    },
    "sample": pd.DataFrame({
        "fixed acidity": [7.4, 7.8, 7.8],
        "volatile acidity": [0.70, 0.88, 0.76],
        "citric acid": [0.00, 0.00, 0.04],
        "residual sugar": [1.9, 2.6, 2.3],
        "chlorides": [0.076, 0.098, 0.092],
        "free sulfur dioxide": [11, 25, 15],
        "total sulfur dioxide": [34, 67, 54],
        "density": [0.9978, 0.9968, 0.9970],
        "pH": [3.51, 3.20, 3.26],
        "sulphates": [0.56, 0.68, 0.65],
        "alcohol": [9.4, 9.8, 9.8],
        "quality": [5, 5, 5],
    }),
},

"churn": {
    "name": "Telco Customer Churn",
    "source": "Kaggle - blastchar/telco-customer-churn",
    "size": "~7043 müşteri kaydı",
    "description": "Telekom müşterilerinin demografik/sözleşme bilgilerine göre şirketi "
                    "terk edip etmeyeceğinin (churn) tahmin edildiği ikili sınıflandırma veri seti.",
    "columns": {
        "gender": "Cinsiyet",
        "SeniorCitizen": "65 yaş üstü mü (0/1)",
        "Partner": "Evli/partner var mı",
        "Dependents": "Bakmakla yükümlü kişi var mı",
        "tenure": "Müşteri kalma süresi (ay)",
        "PhoneService": "Telefon servisi var mı",
        "MonthlyCharges": "Aylık ödeme ($)",
        "TotalCharges": "Toplam ödeme ($)",
        "Churn": "Hedef değişken — Müşteri terk etti mi (Yes/No)",
    },
    "sample": pd.DataFrame({
        "gender": ["Female", "Male", "Male"],
        "SeniorCitizen": [0, 0, 1],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [1, 34, 2],
        "PhoneService": ["No", "Yes", "Yes"],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": [29.85, 1889.5, 108.15],
        "Churn": ["No", "No", "Yes"],
    }),
},

"nba": {
    "name": "NBA Player Stats",
    "source": "Kaggle - NBA Players stats (basketball-reference verisi)",
    "size": "Sezon başına ~500 oyuncu kaydı",
    "description": "Oyuncu başına maç istatistiklerine (sayı, ribaund, asist) göre "
                    "K-Means ile oyun stiline göre kümelenen veri seti.",
    "columns": {
        "Player": "Oyuncu adı",
        "PTS": "Maç başına sayı",
        "REB": "Maç başına ribaund",
        "AST": "Maç başına asist",
    },
    "sample": pd.DataFrame({
        "Player": ["Player A", "Player B", "Player C"],
        "PTS": [24.5, 8.2, 15.1],
        "REB": [4.1, 10.5, 5.6],
        "AST": [6.3, 1.2, 8.9],
    }),
},

"creditcard": {
    "name": "Credit Card Dataset for Clustering",
    "source": "Kaggle - arjunbhasin2013/ccdata",
    "size": "~8950 müşteri kaydı",
    "description": "Kredi kartı kullanıcılarının bakiye, harcama ve limit bilgilerine göre "
                    "K-Means ile segmentlere ayrıldığı kümeleme veri seti.",
    "columns": {
        "BALANCE": "Mevcut hesap bakiyesi ($)",
        "PURCHASES": "Toplam alışveriş tutarı ($)",
        "CREDIT_LIMIT": "Kredi kartı limiti ($)",
    },
    "sample": pd.DataFrame({
        "BALANCE": [40.9, 3202.5, 1500.0],
        "PURCHASES": [95.4, 6442.9, 500.0],
        "CREDIT_LIMIT": [1000.0, 7000.0, 4000.0],
    }),
},

"spotify": {
    "name": "Spotify Tracks Dataset",
    "source": "Kaggle - maharshipandya/-spotify-tracks-dataset",
    "size": "~114.000 şarkı kaydı",
    "description": "Spotify API'sinden alınan ses özelliklerine (danceability, energy, "
                    "loudness vb.) göre şarkıların K-Means ile kümelendiği veri seti.",
    "columns": {
        "track_name": "Şarkı adı",
        "artists": "Sanatçı",
        "danceability": "Dans edilebilirlik (0–1)",
        "energy": "Enerji seviyesi (0–1)",
        "loudness": "Ses seviyesi (dB)",
    },
    "sample": pd.DataFrame({
        "track_name": ["Song A", "Song B", "Song C"],
        "artists": ["Artist X", "Artist Y", "Artist Z"],
        "danceability": [0.65, 0.82, 0.40],
        "energy": [0.71, 0.88, 0.35],
        "loudness": [-6.5, -4.2, -12.1],
    }),
},

"drowsy": {
    "name": "— (Veri seti yok)",
    "source": "—",
    "size": "—",
    "description": "Bu modül gerçek bir veri setiyle eğitilmemiştir; şu an rastgele "
                    "simülasyon çalıştırıyor. Gerçek bir uygulamada MRL Eye Dataset veya "
                    "benzeri açık/kapalı göz görüntü veri setleri kullanılır.",
    "columns": None,
    "sample": None,
    "note": "Bu proje için henüz bir eğitim veri seti tanımlanmadı (simülasyon modu).",
},

"mask": {
    "name": "Face Mask Detection Dataset",
    "source": "Kaggle - andrewmvd/face-mask-detection (veya benzeri with_mask/without_mask klasör yapısı)",
    "size": "~2 sınıf, binlerce yüz görüntüsü",
    "description": "Maskeli ve maskesiz yüz fotoğraflarından oluşan, görüntülerin 64x64 "
                    "boyutuna indirgenip düzleştirilerek (flatten) klasik ML modeline "
                    "verildiği görüntü sınıflandırma veri seti. Tablo verisi değil, "
                    "klasör bazlı görüntü veri setidir.",
    "columns": {
        "with_mask/": "Maskeli yüz görüntüleri klasörü",
        "without_mask/": "Maskesiz yüz görüntüleri klasörü",
    },
    "sample": None,
    "note": "Görsel veri seti olduğu için tablo örneği yerine klasör yapısı gösterilmiştir.",
},

"hand": {
    "name": "— (Veri seti yok, pretrained model)",
    "source": "Google MediaPipe Hands (önceden eğitilmiş)",
    "size": "—",
    "description": "Bu modül kendi veri setiyle eğitilmemiştir; Google'ın milyonlarca el "
                    "görüntüsüyle önceden eğittiği MediaPipe Hands modelini kullanır.",
    "columns": None,
    "sample": None,
    "note": "Eğitim verisi gerekmiyor — pretrained model kullanılıyor.",
},

"sms": {
    "name": "SMS Spam Collection Dataset",
    "source": "UCI Machine Learning Repository / Kaggle - uciml/sms-spam-collection-dataset",
    "size": "5572 SMS mesajı",
    "description": "İngilizce SMS mesajlarının spam veya normal (ham) olarak etiketlendiği, "
                    "TF-IDF + Naive Bayes ile sınıflandırılan metin veri seti.",
    "columns": {
        "v1 (label)": "Etiket — spam / ham",
        "v2 (text)": "Mesaj metni",
    },
    "sample": pd.DataFrame({
        "label": ["ham", "spam", "ham"],
        "text": [
            "Ok lar... Joking wif u oni...",
            "Congratulations! You've won a $1,000 Walmart Gift Card. Click here.",
            "I'll call you later, in meeting.",
        ],
    }),
},

"imdb_sentiment": {
    "name": "IMDb Movie Reviews Dataset",
    "source": "Kaggle - lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
    "size": "50.000 film eleştirisi (25k pozitif / 25k negatif)",
    "description": "Film eleştirilerinin pozitif/negatif olarak etiketlendiği, TF-IDF + "
                    "Logistic Regression ile duygu analizi yapılan metin veri seti.",
    "columns": {
        "review": "Eleştiri metni (İngilizce)",
        "sentiment": "Hedef değişken — positive / negative",
    },
    "sample": pd.DataFrame({
        "review": [
            "The movie was absolutely fantastic! Superb acting.",
            "Waste of time, terrible plot and bad acting.",
        ],
        "sentiment": ["positive", "negative"],
    }),
},

"fake_news": {
    "name": "Fake and Real News Dataset",
    "source": "Kaggle - clmentbisaillon/fake-and-real-news-dataset",
    "size": "~44.000 haber makalesi (True.csv + Fake.csv)",
    "description": "Gerçek ve sahte haber makalelerinin ayırt edilmeye çalışıldığı, "
                    "TF-IDF + Passive Aggressive Classifier ile sınıflandırılan metin veri seti.",
    "columns": {
        "title": "Haber başlığı",
        "text": "Haber içeriği",
        "subject": "Haber kategorisi",
        "date": "Yayın tarihi",
        "label": "Hedef değişken — 0: Gerçek, 1: Sahte",
    },
    "sample": pd.DataFrame({
        "title": ["Government announces new policy", "BREAKING: Aliens land in NYC"],
        "text": ["The government officially confirmed...", "Sources claim unverified..."],
        "label": [0, 1],
    }),
},

"movie_rec": {
    "name": "The Movies Dataset",
    "source": "Kaggle - rounakbanik/the-movies-dataset",
    "size": "~45.000 film kaydı",
    "description": "Film türü ve özet (overview) metinlerinin TF-IDF ile vektörleştirilip "
                    "kosinüs benzerliğiyle içerik tabanlı öneri yapıldığı veri seti.",
    "columns": {
        "title": "Film adı",
        "genres": "Film türleri",
        "overview": "Film özeti",
    },
    "sample": pd.DataFrame({
        "title": ["Movie A", "Movie B", "Movie C"],
        "genres": ["Action Sci-Fi", "Drama Romance", "Action Thriller"],
        "overview": ["A hero saves the world...", "Two people fall in love...", "A detective chases..."],
    }),
},

"book_rec": {
    "name": "Books Dataset (Goodreads benzeri)",
    "source": "Kaggle - jealousleopard/goodreadsbooks (veya benzeri)",
    "size": "Binlerce kitap kaydı",
    "description": "Kitap yazarı ve açıklama metinlerinin TF-IDF ile vektörleştirilip "
                    "kosinüs benzerliğiyle öneri yapıldığı veri seti.",
    "columns": {
        "title": "Kitap adı",
        "authors": "Yazar(lar)",
        "description": "Kitap açıklaması",
    },
    "sample": pd.DataFrame({
        "title": ["Book A", "Book B", "Book C"],
        "authors": ["Author X", "Author Y", "Author X"],
        "description": ["A tale of...", "An epic journey...", "A mystery about..."],
    }),
},

"song_rec": {
    "name": "Spotify Tracks Dataset (ses özellikleri)",
    "source": "Kaggle - maharshipandya/-spotify-tracks-dataset",
    "size": "~114.000 şarkı kaydı",
    "description": "Şarkıların ses özelliklerinin (danceability, energy, valence, tempo, "
                    "loudness) ölçeklenip kosinüs benzerliğiyle öneri yapıldığı veri seti.",
    "columns": {
        "track_name": "Şarkı adı",
        "artists": "Sanatçı",
        "danceability": "Dans edilebilirlik",
        "energy": "Enerji",
        "valence": "Pozitiflik / mutluluk düzeyi",
        "tempo": "Tempo (BPM)",
        "loudness": "Ses seviyesi (dB)",
    },
    "sample": pd.DataFrame({
        "track_name": ["Song A", "Song B"],
        "artists": ["Artist X", "Artist Y"],
        "danceability": [0.65, 0.82],
        "energy": [0.71, 0.88],
        "valence": [0.55, 0.30],
        "tempo": [120.0, 98.5],
        "loudness": [-6.5, -4.2],
    }),
},

"stock": {
    "name": "AAPL Hisse Senedi Verisi",
    "source": "Yahoo Finance (yfinance kütüphanesi ile canlı çekim)",
    "size": "Günlük veri, seçilen tarih aralığına göre değişken",
    "description": "Apple (AAPL) hissesinin geçmiş kapanış fiyatlarından yola çıkarak "
                    "bir sonraki günün fiyatının tahmin edildiği zaman serisi veri seti.",
    "columns": {
        "Date": "İşlem tarihi",
        "Open": "Açılış fiyatı",
        "High": "Gün içi en yüksek",
        "Low": "Gün içi en düşük",
        "Close": "Kapanış fiyatı (özellik olarak kullanılan)",
        "Volume": "İşlem hacmi",
    },
    "sample": pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-04", "2023-01-05"],
        "Close": [125.07, 126.36, 125.02],
        "Volume": [112117500, 89113600, 80962700],
    }),
},

"weather": {
    "name": "Daily Temperature Dataset",
    "source": "NOAA / Kaggle iklim veri setleri (örnek: Delhi/İstanbul günlük sıcaklık verisi)",
    "size": "Günlük veri, yıllara göre değişken",
    "description": "Geçmiş günlük ortalama sıcaklık değerlerinden yola çıkarak gelecekteki "
                    "sıcaklığın tahmin edildiği zaman serisi veri seti.",
    "columns": {
        "date": "Tarih",
        "avg_temp": "Günlük ortalama sıcaklık (°C)",
    },
    "sample": pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "avg_temp": [8.2, 7.9, 9.1],
    }),
},

"walmart": {
    "name": "Walmart Recruiting - Store Sales Forecasting",
    "source": "Kaggle - walmart-recruiting-store-sales-forecasting",
    "size": "~421.000 haftalık satış kaydı, 45 mağaza",
    "description": "Mağazaların haftalık ciro verilerinden yola çıkarak gelecek haftaların "
                    "cirosunun tahmin edildiği zaman serisi veri seti.",
    "columns": {
        "Store": "Mağaza numarası",
        "Date": "Hafta tarihi",
        "Weekly_Sales": "Haftalık ciro ($)",
    },
    "sample": pd.DataFrame({
        "Store": [1, 1, 1],
        "Date": ["2010-02-05", "2010-02-12", "2010-02-19"],
        "Weekly_Sales": [1643690.90, 1641957.44, 1611968.17],
    }),
},

"social_media_viz": {
    "name": "Social Media Usage and Emotional Well-Being Dataset",
    "source": "Kaggle - social medya kullanım alışkanlıkları veri seti",
    "size": "Birkaç bin kullanıcı kaydı",
    "description": "Kullanıcıların yaş, cinsiyet, platform ve günlük kullanım sürelerine göre "
                    "baskın duygu durumlarının görselleştirildiği veri seti (model eğitimi yok).",
    "columns": {
        "Gender": "Cinsiyet",
        "Platform": "Sosyal medya platformu",
        "Daily_Usage_Time (minutes)": "Günlük kullanım süresi (dakika)",
        "Age": "Yaş",
        "Dominant_Emotion": "Baskın duygu durumu",
    },
    "sample": pd.DataFrame({
        "Gender": ["Female", "Male"],
        "Platform": ["Instagram", "TikTok"],
        "Daily_Usage_Time (minutes)": [120, 200],
        "Age": [22, 19],
        "Dominant_Emotion": ["Happy", "Anxious"],
    }),
},

"co2_viz": {
    "name": "Our World in Data - CO2 Emissions",
    "source": "ourworldindata.org/co2-dataset (owid-co2-data.csv)",
    "size": "200+ ülke, 1850'den günümüze yıllık veri",
    "description": "Ülkelerin yıllara göre kömür, petrol ve gaz kaynaklı CO2 emisyonlarının "
                    "karşılaştırıldığı veri seti (model eğitimi yok, sadece görselleştirme).",
    "columns": {
        "Country": "Ülke",
        "Year": "Yıl",
        "Total": "Toplam CO2 emisyonu",
        "Coal": "Kömür kaynaklı emisyon",
        "Oil": "Petrol kaynaklı emisyon",
        "Gas": "Gaz kaynaklı emisyon",
    },
    "sample": pd.DataFrame({
        "Country": ["United States", "China"],
        "Year": [2020, 2020],
        "Total": [4712.8, 10667.9],
        "Coal": [986.1, 7405.2],
        "Oil": [2143.5, 1512.3],
        "Gas": [1583.2, 750.4],
    }),
},

"ecommerce_viz": {
    "name": "Online Retail Dataset",
    "source": "UCI Machine Learning Repository - Online Retail",
    "size": "~540.000 işlem kaydı, İngiltere merkezli e-ticaret",
    "description": "Bir e-ticaret firmasının işlem geçmişinden ciro ve ürün popülerliği "
                    "analizlerinin yapıldığı veri seti (model eğitimi yok).",
    "columns": {
        "InvoiceNo": "Fatura numarası",
        "Description": "Ürün açıklaması",
        "Quantity": "Satılan adet",
        "UnitPrice": "Birim fiyat",
        "CustomerID": "Müşteri numarası",
        "Country": "Ülke",
        "Total_Price": "Hesaplanan toplam tutar (Quantity × UnitPrice)",
    },
    "sample": pd.DataFrame({
        "InvoiceNo": ["536365", "536365"],
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN"],
        "Quantity": [6, 6],
        "UnitPrice": [2.55, 3.39],
        "Country": ["United Kingdom", "United Kingdom"],
    }),
},

"pneumonia": {
    "name": "Chest X-Ray Images (Pneumonia)",
    "source": "Kaggle - paultimothymooney/chest-xray-pneumonia",
    "size": "~5856 göğüs röntgeni görüntüsü (NORMAL / PNEUMONIA)",
    "description": "Göğüs röntgeni görüntülerinden zatürre teşhisi yapan CNN'in eğitildiği "
                    "görüntü veri seti. Tablo verisi değil, klasör bazlı görüntü veri setidir.",
    "columns": {
        "train/NORMAL/": "Sağlıklı akciğer röntgenleri",
        "train/PNEUMONIA/": "Zatürreli akciğer röntgenleri",
    },
    "sample": None,
    "note": "Görsel veri seti olduğu için tablo örneği yerine klasör yapısı gösterilmiştir.",
},

"face_emotion": {
    "name": "FER2013 (Facial Expression Recognition)",
    "source": "Kaggle - msambare/fer2013",
    "size": "~35.000 adet 48x48 gri tonlamalı yüz görüntüsü, 7 duygu sınıfı (bu projede 3'ü kullanılıyor)",
    "description": "Yüz ifadelerinden duygu durumu (öfkeli/mutlu/üzgün) sınıflandıran CNN'in "
                    "eğitildiği görüntü veri seti.",
    "columns": {
        "pixels": "48x48 piksel değerleri (gri tonlama)",
        "emotion": "Hedef değişken — duygu etiketi",
    },
    "sample": None,
    "note": "Görsel/piksel veri seti olduğu için tablo örneği anlamlı değildir.",
},

"text_gen": {
    "name": "Serbest Metin Korpüsü (örn. Shakespeare eserleri)",
    "source": "Örnek: Project Gutenberg - Shakespeare tam metinleri",
    "size": "Metin dosyasının büyüklüğüne bağlı (~100K+ kelime)",
    "description": "Ham metin üzerinden kelime geçiş olasılıklarının (Markov Zinciri) "
                    "çıkarıldığı, yapısal olmayan (unstructured) metin veri seti.",
    "columns": None,
    "sample": pd.DataFrame({
        "örnek_cümle": ["to be or not to be that is the question"]
    }),
    "note": "Tablo verisi değil, düz metin (corpus.txt) kullanılıyor.",
},

"farm_agent": {
    "name": "— (Veri seti yok, kural tabanlı ajan)",
    "source": "—",
    "size": "—",
    "description": "Bu ajan bir veri setiyle eğitilmemiştir; toprak nemi/sıcaklık/güneş "
                    "ışığı eşik değerleri uzman bilgisiyle (domain knowledge) belirlenmiştir.",
    "columns": None,
    "sample": None,
    "note": "Kural tabanlı sistem — eğitim verisi gerekmiyor.",
},

"faq_agent": {
    "name": "— (Veri seti yok, kural tabanlı ajan)",
    "source": "—",
    "size": "—",
    "description": "Anahtar kelime eşleştirmesine dayalı basit bir niyet analizi; "
                    "gerçek üretimde bu bir metin sınıflandırma veri setiyle (örn. etiketli "
                    "müşteri talepleri) değiştirilebilir.",
    "columns": None,
    "sample": None,
    "note": "Kural tabanlı sistem — eğitim verisi gerekmiyor.",
},

"autonomous_car": {
    "name": "— (Veri seti yok, kural tabanlı ajan)",
    "source": "—",
    "size": "—",
    "description": "Sensör mesafesi ve şerit durumu eşiklerine dayalı kural tabanlı bir "
                    "karar mekanizması; gerçek otonom araçlarda bu katman LIDAR/kamera "
                    "verisiyle eğitilmiş bir modelle desteklenir.",
    "columns": None,
    "sample": None,
    "note": "Kural tabanlı sistem — eğitim verisi gerekmiyor.",
},

}
