# 🐶🐱 CNN ile Kedi / Köpek Sınıflandırma

Bu proje **[Akbank Derin Öğrenme Bootcamp](https://www.akbanklab.com/)** kapsamında,  
[Kaggle – Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) veri seti kullanılarak  
**Convolutional Neural Network (CNN)** ile **kedi ve köpek görsellerinin sınıflandırılması** ve  
**Grad-CAM** ile modelin karar bölgelerinin görselleştirilmesi üzerine hazırlanmıştır.

---

## 📂 İçindekiler
- [Kurulum](#kurulum)
- [Veri Seti](#veri-seti)
- [Model Mimarisi](#model-mimarisi)
- [Eğitim Süreci](#eğitim-süreci)
- [Sonuçlar](#sonuçlar)
- [Grad-CAM Görselleştirme](#grad-cam-görselleştirme)
- [Değerlendirme ve Gelecek Çalışmalar](#değerlendirme-ve-gelecek-çalışmalar)
- [Lisans](#lisans)

---

## 💻 Kurulum

Projeyi yerel makinenize klonlayın ve gerekli kütüphaneleri kurun:

```bash
git clone [https://github.com/<kullanici_adi>/dogs-vs-cats-cnn.git](https://github.com/HumeyraErtas/dogs-vs-cats-cnn)
cd dogs-vs-cats-cnn
pip install -r requirements.txt
requirements.txt dosyasında temel kütüphaneler:

tensorflow

scikit-learn

numpy

matplotlib

pillow

🗂 Veri Seti
Kaynak: Kaggle – Dogs vs Cats   (https://www.kaggle.com/code/hmeyra/dogs-vs-cats-cnn)

Toplam Görsel: 25.000 (12.500 kedi – 12.500 köpek)

Eğitim/Doğrulama Bölünmesi: %80 / %20

Veri Artırma: ImageDataGenerator ile döndürme, zoom, yatay çevirme uygulanmıştır.

🧩 Model Mimarisi
Aşağıdaki CNN mimarisi ile ikili sınıflandırma yapılmıştır:

scss
Kodu kopyala
Conv2D(32) → ReLU → MaxPooling
Conv2D(64) → ReLU → MaxPooling
Conv2D(128) → ReLU → MaxPooling
Flatten
Dense(512) → ReLU
Dropout(0.5)
Dense(1) → Sigmoid
Kayıp fonksiyonu: Binary Crossentropy

Optimizasyon: Adam

Öğrenme Oranı Denemeleri: 0.0001 ve 0.001

🏋️‍♀️ Eğitim Süreci
Kısa Test (3 Epoch)
Learning Rate	Epoch	Train Accuracy	Val Accuracy
0.0001	3	%70.8	%73.6
0.001	3	%70.5	%74.5

0.001 öğrenme oranı daha yüksek doğruluk sağladığı için tam eğitimde tercih edilmiştir.

Tam Eğitim (15 Epoch)
Son Epoch (15/15):

Eğitim doğruluğu: %82.4

Doğrulama doğruluğu: %84.36

Doğrulama kaybı: 0.369

Model 10. epoch sonrası istikrarlı biçimde %80+ doğruluk seviyelerine ulaşmıştır.

✅ Sonuçlar
Doğrulama Seti Başarısı: %84 civarı

Erken test sırasında (3 epoch) 5000 test görseli için elde edilen sınıflandırma raporu:

yaml
Kodu kopyala
precision    recall  f1-score   support

0 (cat)      0.51    0.51       0.51     2500
1 (dog)      0.51    0.50       0.50     2500

accuracy                         0.51    5000
macro avg    0.51    0.51       0.51     5000
weighted avg 0.51    0.51       0.51     5000
Not: Bu skorlar erken epoch sonuçlarıdır. 15 epoch tam eğitim sonunda doğruluk %84’e yükselmiştir.

🔥 Grad-CAM Görselleştirme
Modelin karar verirken hangi bölgelere odaklandığını görmek için Grad-CAM uygulanmıştır:

python
Kodu kopyala
heatmap, overlay = grad_cam(model, "ornek_resim.jpg")
plt.imshow(overlay)
📷 Gözlem: Model, genellikle hayvanın yüz ve gövde bölgelerine yoğunlaşmaktadır.

📊 Değerlendirme ve Gelecek Çalışmalar
Confusion Matrix ve Classification Report scikit-learn ile hesaplanmıştır.

Veri artırma stratejileri ve daha derin mimarilerle doğruluk daha da artırılabilir.

Transfer learning (ör. ResNet, VGG16) ile daha yüksek doğruluk elde edilebilir.

📄 Lisans
Bu proje MIT Lisansı altında yayımlanmıştır.

👩‍💻 Katkı
Pull request ve issue’lar her zaman açıktır.
Geliştirme önerilerinizi paylaşabilirsiniz.
