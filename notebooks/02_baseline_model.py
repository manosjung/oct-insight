from fastai.vision.all import *
import os

if __name__ == "__main__":
    # --- AYARLAR ---
    # Veri yolunu tanımlıyoruz
    path = Path("data/raw/OCT2017")
    # Modellerin kaydedileceği klasör
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    print("--- Model Eğitimi Başlıyor ---")

    # 1. Veri Yükleme (DataBlock Oluşturma)
    # FastAI'ın "DataBlock" yapısı, veriyi nasıl okuyacağını tarif ettiğimiz yerdir.
    print("1. Veriler hazırlanıyor...")
    octs = DataBlock(
        blocks=(ImageBlock, CategoryBlock), # Girdi: Resim, Çıktı: Kategori
        get_items=get_image_files,          # Resim dosyalarını bulma fonksiyonu
        splitter=GrandparentSplitter(train_name='train', valid_name='test'), # Klasör isimlerine göre eğitim/test ayrımı
        get_y=parent_label,                 # Etiketi klasör isminden al (CNV, NORMAL vs.)
        item_tfms=Resize(224),              # Resimleri 224x224 boyutuna getir (ResNet için standart)
        batch_tfms=aug_transforms(size=224, min_scale=0.75) # Veri çoğaltma (Augmentation) - döndürme, yakınlaştırma vb.
    )

    # DataLoader'ı oluşturuyoruz. Bu, verileri gruplar (batch) halinde modele besler.
    # batch_size=16 yaptık, bilgisayarın hafızası yetmezse düşürebiliriz (16 veya 8).
    dls = octs.dataloaders(path, batch_size=64, num_workers=0)

    print(f"   Eğitim setindeki resim sayısı: {len(dls.train_ds)}")
    print(f"   Doğrulama (Test) setindeki resim sayısı: {len(dls.valid_ds)}")
    print(f"   Sınıflar: {dls.vocab}")

    # 2. Modeli Oluşturma (Learner)
    # vision_learner fonksiyonu ile hazır bir model (ResNet50) kullanıyoruz.
    # metrics=accuracy ile başarımızı "doğruluk oranı" olarak göreceğiz.
    print("2. ResNet50 modeli indiriliyor ve hazırlanıyor...")
    learn = vision_learner(dls, resnet50, metrics=accuracy, path=Path("."))

    # 3. Eğitim (Training)
    # fine_tune: Transfer learning için kullanılan sihirli fonksiyon.
    # Önce son katmanı eğitir, sonra tüm ağı yavaşça eğitir.
    # epochs=1 yaptık şimdilik, çünkü veri çok büyük, uzun sürebilir.
    print("3. Eğitim başlıyor (Bu işlem bilgisayar hızına göre zaman alabilir)...")
    learn.fine_tune(1)

    # 4. Kaydetme
    print("4. Model kaydediliyor...")
    learn.export('models/baseline_model.pkl')
    print(f"   Model şuraya kaydedildi: {model_dir}/baseline_model.pkl")

    print("--- İşlem Tamamlandı ---")
