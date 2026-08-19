# 🚀 10 Kategoride 30 Projelik Makine Öğrenmesi & Yapay Zeka Portfolyosu

Bu proje, veri bilimi ve yapay zeka alanındaki temel ve ileri düzey metodolojileri içeren, **10 farklı kategoride toplam 30 adet** projenin tek bir Streamlit arayüzünde birleştirildiği kapsamlı bir akademik portfolyo çalışmasıdır.

Uygulama üzerindeki tüm modeller **Kaggle API** aracılığıyla gerçek veri setleri kullanılarak eğitilmiş, optimize edilmiş ve canlıya alınmıştır.

---

## 🗺️ Portfolyo Genel Mimarisi ve Projeler

Uygulama sol tarafta yer alan dinamik bir menü üzerinden kontrol edilmektedir ve şu 10 ana kategoriyi/projeleri içerir:

### 1. 📊 Regresyon (Regression)
*   **Altın Fiyatı Tahmini:** Küresel finansal endeksleri (S&P 500, Petrol, Gümüş, EUR/USD) kullanarak altın fiyat trendini tahmin eder.
*   **Öğrenci Sınav Puanı Tahmini:** Sosyal ve eğitsel demografi verilerine dayanarak öğrencinin matematik notunu öngörür.
*   **Uber/Taksi Ücret Tahmini:** Canlı harita entegrasyonu ile enlem/boylam ve yolcu sayısına göre yolculuk maliyetini hesaplar.

### 2. 🔵 Sınıflandırma (Classification)
*   **Mobil Cihaz Fiyat Segmenti:** Telefonların donanımsal özelliklerine (RAM, Batarya vb.) göre hangi fiyat sınıfına ait olduğunu bulur.
*   **Şarap Kalitesi Sınıflandırması:** Kimyasal ölçümlere (pH, alkol, asitlik) göre şarap kalitesini puanlar.
*   **Müşteri Terki (Churn) Tahmini:** Telekomünikasyon müşterilerinin kullanım alışkanlıklarına göre şirketten ayrılma riskini hesaplar.

### 3. 🟡 Kümeleme (Clustering)
*   **NBA Oyuncu Performans Gruplaması:** Maç başına istatistiklere göre oyuncuları otonom oyun stillerine ayırır (3D Görselleştirmeli).
*   **Kredi Kartı Müşteri Segmentasyonu:** Banka müşterilerinin harcama limitleri ve bakiyelerine göre finansal profiller oluşturur.
*   **Spotify Şarkı Tarzı Kümeleme:** Şarkıların dans edilebilirlik ve enerji gibi müzikal karakterlerine göre atmosfer gruplaması yapar.

### 4. 📷 Bilgisayarlı Görü (Computer Vision)
*   **Sürücü Uyuklama Tespiti:** Göz açıklık oranını (EAR) analiz ederek sürücü yorgunluk durumunu tespit eder.
*   **Maske Kullanımı Tespiti:** Yüklenen fotoğraflardaki yüzleri tarayarak maskeli/maskesiz ayrımı yapar.
*   **El İşaretleri ve Parmak Sayma:** `MediaPipe` kütüphanesi ile el eklem noktalarını çıkartarak açık parmak sayısını hesaplar.

### 5. 📝 Doğal Dil İşleme (NLP)
*   **SMS Spam / Kimlik Avı Tespiti:** Gelen metinleri TF-IDF ile analiz ederek dolandırıcılık veya spam girişimlerini saptar.
*   **IMDb Yorumları Duygu Analizi:** Film eleştirilerinin arkasındaki duygu durumunun olumlu mu yoksa olumsuz mu olduğunu sınıflandırır.
*   **Sahte Haber Tespiti:** Haber metinlerinin doğruluğunu ve manipülasyon riskini analiz eder.

### 6. 🍿 Öneri Sistemleri (Recommendation Systems)
*   **IMDb Film Öneri Sistemi:** Tür ve özet benzerliklerine (Cosine Similarity) göre film tavsiye eder.
*   **Kitap Tavsiye Motoru:** Yazar ve içerik ortaklıklarına göre okuma listesi önerileri sunar.
*   **Şarkı / Müzik Öneri Sistemi:** Spotify ritim ve tarz benzerliklerine göre şarkı kuyruğu oluşturur.

### 7. 📈 Zaman Serileri (Time Series)
*   **Hisse Senedi Fiyat Tahmini:** Apple (AAPL) hissesinin geçmiş verilerinden yararlanarak gelecek günlerin trendini çizer.
*   **Hava Durumu / Sıcaklık Tahmini:** Tarihsel iklim verileriyle önümüzdeki günlerin ortalama sıcaklık grafiğini öngörür.
*   **Mağaza Satış Tahmini:** Walmart haftalık satış verilerini analiz ederek gelecek dönem ciro talebini hesaplar.

### 8. 📊 Veri Görselleştirme (Data Viz)
*   **Küresel Sosyal Medya İstatistikleri:** Platform bazlı günlük kullanım sürelerini ve yaş gruplarının duygu dağılımlarını gösterir.
*   **İklim Değişikliği & Karbon Salınımı:** Ülkelerin fosil yakıt kaynaklı CO2 emisyon trendlerini interaktif olarak kıyaslar.
*   **E-Ticaret Satış Dashboard:** Online mağazanın toplam cirosunu, ülke dağılımlarını ve en çok satan ürünlerini grafikleştirir.

### 9. 🧠 Derin Öğrenme (Deep Learning)
*   **Göğüs Röntgeninden Zatürre Teşhisi:** X-Ray görsellerini Evrişimli Sinir Ağları (CNN) ile tarayarak zatürre (Pneumonia) teşhisi koyar.
*   **Yüz İfadesinden Duygu Tanıma:** İnsan yüzündeki mimiklerden baskın duygu durumunu (Öfkeli, Mutlu, Üzgün) tahmin eden CNN modelidir.
*   **Metin Üretim Robotu:** Karakter/Kelime tabanlı olasılık zincirleri ile verilen kelimeden otonom metinler üretir.

### 10. 🤖 Yapay Zeka Ajanları (AI Agents)
*   **Akıllı Tarım ve Sulama Ajanı:** Toprak nemi ve buharlaşma riskine göre sulama kararlarını otonom alan karar ağacı ajanı.
*   **SSS Destek ve Niyet Analizi Ajanı:** Müşteri mesajlarının arkasındaki gerçek niyeti çözerek otonom departman yönlendirmesi yapan ajan.
*   **Otonom Araç Simülasyon Ajanı:** Sensör verilerine göre şerit takip, acil fren ve otonom park kararlarını simüle eden araç ajanı.

---

## 🛠️ Kurulum ve Yerelde Çalıştırma

Projenizi kendi bilgisayarınızda test etmek için aşağıdaki adımları takip edebilirsiniz:

1. Bu depoyu bilgisayarınıza indirin:
   ```bash
   git clone https://github.com
   cd REPO_ADINIZ
   ```

2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Modelleri Kaggle üzerinden eğitip `models/` klasörüne kaydetmek için eğitim scriptlerini çalıştırın (Örn):
   ```bash
   python train_regression.py
   python train_classification.py
   # (Tüm kategorilerin train scriptleri çalıştırılmalıdır)
   ```

4. Streamlit uygulamasını başlatın:
   ```bash
   streamlit run app.py
   ```

---

## 📦 Proje Dosya Yapısı

```text
├── 📄 app.py                     # Ana Streamlit arayüz kodları
├── 📄 requirements.txt           # Bağımlı kütüphaneler listesi
├── 📄 README.md                  # Proje açıklama dokümanı
├── 📁 models/                    # Eğitilmiş tüm .pkl, .scaler ve .h5 dosyaları
└── 📁 scripts/                   # Modelleri eğiten train_...py kodları
```

## 📜 Teknolojiler ve Kütüphaneler
* **Arayüz & Dashboard:** Streamlit, Plotly Express
* **Makine Öğrenmesi & NLP:** Scikit-Learn, Joblib, Naive Bayes
* **Derin Öğrenme & CV:** TensorFlow, Keras, OpenCV, MediaPipe
* **Veri Yönetimi:** Pandas, NumPy
* **Veri Kaynağı:** Kaggle Hub API