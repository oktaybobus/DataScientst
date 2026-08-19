import json, uuid, ast

SRC = '/home/ubuntu/Uploads/train_datasci.ipynb'
OUT = '/home/ubuntu/train_datasci_final.ipynb'

with open(SRC, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def get_src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s

def md_cell(text):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }

def code_cell(text):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }

# ============ İŞ 1: Şarap modelini düzelt (idx 18) ============
wine = cells[18]
new_wine = '''from sklearn.ensemble import RandomForestClassifier

try:
    print("🍷 Şarap Kalitesi Modeli Eğitiliyor...")
    wine_path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
    wine_df = pd.read_csv(os.path.join(wine_path, "winequality-red.csv"))
    explore_dataset(wine_df, "Kırmızı Şarap Kalitesi")

    X_wine = wine_df.drop(['quality'], axis=1)
    y_wine = wine_df['quality']

    X_tr_w, X_te_w, y_tr_w, y_te_w = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

    param_grid_wine = {'n_estimators': [200, 300], 'max_depth': [None, 20, 30], 'min_samples_leaf': [1, 2]}
    search_wine = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid_wine,
                                      n_iter=5, cv=5, random_state=42, n_jobs=-1)
    search_wine.fit(X_tr_w, y_tr_w)
    wine_model = search_wine.best_estimator_
    joblib.dump(wine_model, 'models/wine_model.pkl')

    y_pred_wine = wine_model.predict(X_te_w)
    add_classification_metrics("wine", y_te_w, y_pred_wine)
    metrics["wine"]["CV_Accuracy"] = round(float(search_wine.best_score_), 4)
    metrics["wine"]["best_params"] = search_wine.best_params_
    print(f"   📊 Accuracy={metrics['wine']['Accuracy']}  F1={metrics['wine']['F1']}")
    print(f"   🔧 En iyi parametreler: {search_wine.best_params_}  (5-fold CV Accuracy={metrics['wine']['CV_Accuracy']})")
    show_feature_importance(wine_model, X_wine.columns)
    show_classification_report("Şarap Kalitesi", y_te_w, y_pred_wine)
    print("✅ Şarap kalitesi modeli kaydedildi: models/wine_model.pkl\\n")
except Exception as e:
    add_error_metric("wine", e)
'''
wine['source'] = new_wine.splitlines(keepends=True)
wine['outputs'] = []
wine['execution_count'] = None
assert 'RandomForestClassifier' in get_src(wine) and 'GradientBoosting' not in get_src(wine)

# ============ İŞ 2: Açıklama markdown'ları ============
# proje alt-başlık indeksi -> açıklama metni
def expl(amac, veri, model, deger):
    return (f"> **📌 Amaç:** {amac}  \n"
            f"> **📂 Veri Seti:** {veri}  \n"
            f"> **🧠 Model:** {model}  \n"
            f"> **📊 Değerlendirme:** {deger}")

explanations = {
 8:  expl("GLD altın ETF fiyatını SPX, USO, SLV ve EUR/USD göstergelerinden tahmin etmek.","Kaggle `gold-price-data`","RandomForestRegressor + RandomizedSearchCV","R², RMSE, MAE"),
 10: expl("Öğrencinin matematik sınav puanını demografik ve hazırlık faktörlerinden tahmin etmek.","`StudentsPerformance`","RandomForestRegressor + RandomizedSearchCV","R², RMSE, MAE"),
 12: expl("Uber/taksi yolculuk ücretini kalkış-varış koordinatları ve yolcu sayısından tahmin etmek.","`uber-fares-dataset` (aykırı değer/outlier temizliği uygulanmıştır)","RandomForestRegressor + RandomizedSearchCV","R², RMSE, MAE"),
 15: expl("Telefonun donanım özelliklerinden fiyat segmentini (0–3) sınıflandırmak.","`mobile-price-classification`","RandomForestClassifier + RandomizedSearchCV","Accuracy, F1"),
 17: expl("Kimyasal özelliklerden kırmızı şarap kalite puanını (3–8) sınıflandırmak.","`red-wine-quality`","RandomForestClassifier (`class_weight='balanced'`)","Accuracy, F1"),
 19: expl("Telekom müşterisinin abonelikten ayrılıp ayrılmayacağını (churn) tahmin etmek.","`telco-customer-churn` (tüm kategorik özellikler one-hot kodlanmıştır)","RandomForestClassifier (`class_weight='balanced'`)","Accuracy, F1"),
 22: expl("Oyuncuları sayı-ribaund-asist istatistiklerine göre performans gruplarına ayırmak.","`nba-players-data`","KMeans (silhouette ile otomatik k seçimi)","Silhouette"),
 24: expl("Kart kullanıcılarını harcama/bakiye davranışlarına göre segmentlere ayırmak.","`ccdata`","KMeans (otomatik k seçimi)","Silhouette"),
 26: expl("Şarkıları ses özelliklerine (danceability, energy, loudness) göre kümelemek.","`spotify-tracks-dataset`","KMeans (otomatik k seçimi)","Silhouette"),
 29: expl("Yüz görüntüsünden maske takılı olup olmadığını sınıflandırmak.","`face-mask-dataset`","RandomForestClassifier (64×64 piksel düzleştirilmiş)","Accuracy, F1"),
 32: expl("SMS metninin spam mı yoksa normal (ham) mı olduğunu tespit etmek.","`sms-spam-collection`","MultinomialNB + TF-IDF","Accuracy, F1"),
 34: expl("Film yorumunun pozitif mi negatif mi olduğunu analiz etmek.","`imdb-50k-reviews`","LogisticRegression + TF-IDF","Accuracy, F1"),
 36: expl("Haber metninin gerçek mi sahte mi olduğunu tespit etmek.","`fake-and-real-news`","LogisticRegression + TF-IDF","Accuracy, F1"),
 39: expl("İçerik tabanlı film önerisi (özet + tür benzerliği).","`tmdb-5000-movies`","TF-IDF + Cosine Similarity","Metrik yok (unsupervised)"),
 41: expl("Başlık + yazar benzerliğine dayalı kitap önerisi.","`goodreadsbooks`","TF-IDF + Cosine Similarity","Metrik yok (unsupervised)"),
 43: expl("Şarkı adı + sanatçı + tür benzerliğine dayalı müzik önerisi.","`spotify-tracks-dataset`","TF-IDF + Cosine Similarity","Metrik yok (unsupervised)"),
 46: expl("AAPL kapanış fiyatını bir önceki günün fiyatından (lag-1) tahmin etmek.","`yfinance` (son 5 yıl)","LinearRegression — **naif persistence modeli**","R², RMSE, MAE"),
 48: expl("Delhi günlük ortalama sıcaklığını lag-1 ile tahmin etmek.","`daily-climate-time-series`","LinearRegression (lag-1)","R², RMSE, MAE"),
 50: expl("Haftalık mağaza satışını lag-1 ile tahmin etmek.","**SENTETİK** (rastgele üretilmiş veri)","LinearRegression (lag-1)","R², RMSE, MAE"),
 53: expl("Sosyal medya kullanım süresi ve platform dağılımını görselleştirmek.","**SENTETİK**","Model yok — yalnızca görselleştirme","—"),
 55: expl("Ülke bazlı CO₂ salınımı trendlerini görselleştirmek.","**SENTETİK**","Model yok — yalnızca görselleştirme","—"),
 57: expl("E-ticaret ürün satış işlemleri ve müşteri demografisini görselleştirmek.","**SENTETİK**","Model yok — yalnızca görselleştirme","—"),
 60: expl("Göğüs röntgeninden zatürre (pneumonia) teşhisi koymak.","`chest-xray-pneumonia`","CNN (2 evrişim bloğu + BatchNorm + Dropout, veri artırma/augmentation ile)","Accuracy, Val_Accuracy, Loss"),
 62: expl("Yüz ifadesinden duygu (kızgın/mutlu/üzgün) tanımak.","`fer2013`","CNN (3 evrişim bloğu + augmentation)","Accuracy, Val_Accuracy, Loss"),
 64: expl("Markov zinciri ile kelime bazlı üretken metin oluşturmak (Shakespeare metni).","Gömülü Shakespeare metni","Markov Chain","Metrik yok (üretken model)"),
}

# yeni cell listesi kur, ilgili başlıklardan sonra açıklama enjekte et
new_cells = []
added = 0
for i, c in enumerate(cells):
    new_cells.append(c)
    if i in explanations:
        new_cells.append(md_cell(explanations[i]))
        added += 1

# ============ İŞ 3: Tüm modelleri indirme (en sona) ============
dl_md = md_cell(
"## ***11-Tüm Modelleri İndirme***\n\n"
"Aşağıdaki hücre, eğitilen tüm modelleri, ölçekleyicileri, vektörleştiricileri ve "
"metrik dosyalarını tek bir `models_bundle.zip` arşivinde toplar. "
"Google Colab'da otomatik indirme başlar; yerel Jupyter'da dosya yolu yazdırılır."
)
dl_code = code_cell(
'''# ====================================================================
# 📦 TÜM MODELLERİ TEK ZIP DOSYASINDA İNDİRME
# Eğitilen tüm modeller, ölçekleyiciler, vektörleştiriciler ve metrik
# dosyaları 'models_bundle.zip' içinde toplanır.
# ====================================================================
import shutil
import os

zip_path = 'models_bundle'
shutil.make_archive(zip_path, 'zip', 'models')
zip_file = zip_path + '.zip'

boyut_mb = os.path.getsize(zip_file) / (1024 * 1024)
dosya_sayisi = len(os.listdir('models'))
print(f"✅ Arşiv oluşturuldu: {zip_file}")
print(f"   📁 {dosya_sayisi} dosya  |  💾 {boyut_mb:.2f} MB")

# Google Colab ortamındaysa otomatik indirme başlat
try:
    from google.colab import files
    files.download(zip_file)
    print("⬇️ İndirme başladı (Colab).")
except ImportError:
    print(f"ℹ️ Yerel ortam: dosyayı şu yoldan indirebilirsiniz -> {os.path.abspath(zip_file)}")
''')
new_cells.append(dl_md)
new_cells.append(dl_code)

nb['cells'] = new_cells

# ============ Kaydet ve doğrula ============
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# doğrulama
with open(OUT, encoding='utf-8') as f:
    check = json.load(f)

code_ok = 0
for c in check['cells']:
    if c['cell_type'] == 'code':
        try:
            ast.parse(''.join(c['source']))
            code_ok += 1
        except SyntaxError as e:
            print("❌ Syntax hatası:", e)

print(f"✅ Kaydedildi: {OUT}")
print(f"   Eklenen açıklama hücresi: {added}")
print(f"   Eklenen indirme hücresi: 2 (1 markdown + 1 code)")
print(f"   Toplam hücre: {len(cells)} -> {len(new_cells)}")
print(f"   Syntax kontrolünden geçen code hücresi: {code_ok}")
print(f"   Şarap modeli: RandomForestClassifier(class_weight='balanced') ✅")
