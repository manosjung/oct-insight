# Geliştirme Günlüğü (Development Log)

Bu dosya, projemizdeki ilerlemeyi, öğrendiğimiz kavramları ve aldığımız notları adım adım kaydettiğimiz yerdir.

## 1. Başlangıç ve Planlama
- **Tarih:** 27 Aralık 2025
- **Durum:** `project_plan.md` dosyasını inceledik ve yol haritamızı belirledik.
- **Hedef:** OCT taramalarından 4 farklı retina hastalığını (CNV, DME, DRUSEN, NORMAL) teşhis eden ve Grad-CAM ile "nereye baktığını" gösteren bir yapay zeka modeli geliştirmek.
- **Teknolojiler:** PyTorch, FastAI, Streamlit.

## 2. Kurulum ve Düzenleme
- **Klasör Yapısı:** `OCT2017` klasörünü `data/raw/OCT2017` konumuna taşıdık. Bu sayede proje klasörümüz daha temiz oldu.
- **Kütüphaneler:** `requirements.txt` içindeki kütüphaneler yüklendi.

## 3. Veri Keşfi (Data Exploration)
- **Toplam Resim Sayısı:** 84,484
- **Eğitim Seti (Train):** 83,484 resim
    - **CNV:** 37,205 (En çok veri bunda var)
    - **NORMAL:** 26,315
    - **DME:** 11,348
    - **DRUSEN:** 8,616 (En az veri bunda var)
    - *Not:* Veri setinde dengesizlik (imbalance) var. CNV sınıfı Drusen'den 4 kat daha fazla. Bunu eğitim sırasında dikkate almalıyız.
- **Test Seti:** 1,000 resim (Her sınıftan 250 tane - bu harika, test sonuçlarımız adil olacak).

## Notlar
- **CNV (Koroit Neovaskülarizasyonu):** Acil tedavi gerektiren "Yaş Tip" sarı nokta hastalığı.
- **Drusen:** "Kuru Tip" sarı nokta hastalığı, takip gerektirir.
- **DME (Diyabetik Maküler Ödem):** Şeker hastalığına bağlı ödem.
- **Normal:** Sağlıklı göz.
