import os
import matplotlib.pyplot as plt

# Veri setinin yeni konumu
base_dir = r"data/raw/OCT2017"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

categories = ["CNV", "DME", "DRUSEN", "NORMAL"]

print("--- Veri Seti Özeti ---")
print(f"Veri Klasörü: {base_dir}")

def count_images(directory, set_name):
    print(f"\n[{set_name} Seti]")
    total = 0
    counts = []
    if not os.path.exists(directory):
        print(f"HATA: {directory} bulunamadı!")
        return [], 0
        
    for category in categories:
        path = os.path.join(directory, category)
        try:
            num_files = len(os.listdir(path))
            print(f"  - {category}: {num_files} resim")
            counts.append(num_files)
            total += num_files
        except FileNotFoundError:
            print(f"  - {category}: KLASÖR YOK")
            counts.append(0)
            
    print(f"  TOPLAM: {total} resim")
    return counts, total

train_counts, train_total = count_images(train_dir, "Eğitim (Train)")
test_counts, test_total = count_images(test_dir, "Test")

# Basit bir grafik çizelim (Eğer çalıştırabilirsek)
try:
    plt.figure(figsize=(10, 5))
    plt.bar(categories, train_counts, color=['red', 'blue', 'orange', 'green'])
    plt.title(f"Eğitim Verisi Dağılımı (Toplam: {train_total})")
    plt.xlabel("Hastalık Kategorisi")
    plt.ylabel("Resim Sayısı")
    plt.savefig("data_distribution.png")
    print("\nGrafik kaydedildi: data_distribution.png")
except Exception as e:
    print(f"\nGrafik çizilemedi: {e}")
