##  Kurulum ve Çalıştırma

Projeyi çalıştırmak için sırasıyla aşağıdaki adımları uygulayın.

---

### **1. Gereksinimleri Yükle**

Gerekli kütüphaneleri kurmak için terminalde aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```
### **2. Modeli Eğit (Pipeline)**
Veri setini indirmek, işlemek ve modeli eğitip kaydetmek için eğitim dosyasını çalıştırın.
Bu işlem artifacts/ klasörüne gerekli model dosyalarını oluşturacaktır.
Modeli eğitmek için terminalde aşağıdaki komutu çalıştırın:

```bash
python train_pipeline.py
```
### **3. Arayüzü Başlat (Streamlit)**

Eğitim bittikten sonra web arayüzünü açmak için:

```bash
streamlit run app.py
```
