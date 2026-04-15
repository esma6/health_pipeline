# 🏥 Multi-Agent Clinical Decision Support Pipeline
### *Çoklu Ajan Büyük Sağlık Verisi İşleme ve Klinik Karar Destek Boru Hattı*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-orange?logo=apachespark)](https://spark.apache.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.x-green)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Llama%203.2%203B-purple)](https://ollama.ai)
[![MIMIC-IV](https://img.shields.io/badge/Dataset-MIMIC--IV-red)](https://physionet.org/content/mimiciv/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Yüksek Lisans Ders Projesi** — Büyük Veri ve Teknolojileri Dersi, 2025–2026

---

## 📌 Proje Özeti

Bu proje, MIMIC-IV gerçek hastane veritabanındaki milyonlarca lab sonucunu **Apache Spark** ile işleyip, **3 adet LLM ajanı** (LangGraph + Ollama/Llama 3.2 3B) kullanarak her hasta için otomatik klinik rapor üreten uçtan uca bir pipeline'dır.

**Temel özellikler:**
- 🔥 Apache Spark ile 3 GB+ büyüklüğündeki CSV verisi işleme
- 🤖 3 zincirleme LLM ajanı: Analiz → Karar → Raporlama
- 📊 MIMIC-IV ground truth ile doğrulanmış sonuçlar (F1: **0.932**)
- 💻 Tamamen yerel çalışır — API ücreti yok, internet bağlantısı gerekmez
- 🧪 16 GB RAM  altında çalışacak şekilde optimize edilmiş

---

## 📊 Sonuçlar

| Metrik | Değer |
|--------|-------|
| **Precision** | 1.000 (Yanlış pozitif = 0) |
| **Recall** | 0.873 |
| **F1 Skoru** | **0.932** |
| **Urgency–Mortalite Pearson Korelasyonu** | 0.244 |
| **Pipeline Hata Oranı** | %0.0 |
| **Ortalama Vaka Süresi** | 131.2 saniye |
| **İşlenen Hasta** | 67 vaka |

### Ground Truth: Urgency vs Mortalite

```
Yüksek aciliyet (≥4) → Ölüm oranı: %16.7
Düşük aciliyet (<4)  → Ölüm oranı:  %0.0
```
> Model yüksek aciliyet skoru verdiğinde gerçekten yüksek ölüm oranı gözlemleniyor — klinik anlamlılık doğrulandı.

---

## 🏗️ Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────┐
│                     KATMAN 1: SPARK                      │
│                                                         │
│  labevents.csv (3GB)  ──►  Anomali Filtresi  ──►  Parquet│
│  patients.csv         ──►  Join İşlemi        ──►  Parquet│
│                                                         │
│  ⚡ 107,727 satır okundu → 21,389 anomali tespit edildi │
└────────────────────────┬────────────────────────────────┘
                         │ Spark kapanır, RAM serbest kalır
┌────────────────────────▼────────────────────────────────┐
│                   KATMAN 2: LangGraph                    │
│                                                         │
│  Parquet ──► [A1: Analiz] ──► [A2: Karar] ──► [A3: Rapor]│
│               Anormal         SOFA/qSOFA     Klinik      │
│               bulgular        aciliyet       özet metin  │
│               tespiti         skoru (1–5)               │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Klasör Yapısı

```
health_pipeline/
├── data/
│   ├── raw/
│   │   ├── hosp/
│   │   │   ├── patients.csv
│   │   │   ├── admissions.csv
│   │   │   ├── labevents.csv       (~3 GB)
│   │   │   ├── d_labitems.csv
│   │   │   └── diagnoses_icd.csv
│   │   └── icu/
│   │       ├── chartevents.csv     (~4 GB)
│   │       └── d_items.csv
│   └── processed/
│       ├── lab_anomalies.parquet
│       └── patient_anomalies.parquet
├── src/
│   ├── spark_pipeline.py       # Katman 1: Büyük veri işleme
│   ├── agents.py               # Katman 2: Ajan tanımları
│   ├── langgraph_pipeline.py   # Katman 2: Durum makinesi
│   └── evaluation.py           # Metrik hesaplama
├── notebooks/
│   └── exploration.ipynb
├── results/
│   ├── spark_benchmark.json
│   ├── llm_results.json
│   └── all_metrics.json
└── requirements.txt
```

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.10+
- Java 11+ (Apache Spark için)
- Ollama (yerel LLM sunucusu)
- 16 GB RAM (önerilen)
- MIMIC-IV veri erişimi (PhysioNet)

### 1. Ortam Kurulumu

```bash
# Repoyu klonla
git clone https://github.com/KULLANICI_ADIN/health_pipeline.git
cd health_pipeline

# Sanal ortam oluştur ve aktifleştir
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Bağımlılıkları kur
pip install -r requirements.txt
```

### 2. Ollama ve Model Kurulumu

```bash
# Ollama'yı kur: https://ollama.ai
ollama pull llama3.2:3b

# Modelin çalıştığını doğrula
ollama run llama3.2:3b "Merhaba!"
```

### 3. MIMIC-IV Verisi

MIMIC-IV erişimi için PhysioNet hesabı ve CITI eğitimi gereklidir:
1. [physionet.org](https://physionet.org) üzerinden kayıt ol
2. CITI eğitimini tamamla
3. MIMIC-IV erişim talebinde bulun
4. Onaylandıktan sonra verileri `data/raw/` altına yerleştir

> **Demo için:** [MIMIC-IV Demo](https://physionet.org/content/mimic-iv-demo/2.2/) veri setini CITI gerektirmeden kullanabilirsiniz.

### 4. Pipeline'ı Çalıştır

```bash
# Adım 1: Spark ile büyük veriyi işle (3-8 dakika)
python src/spark_pipeline.py

# Adım 2: LangGraph ajanlarını çalıştır
python src/langgraph_pipeline.py

# Adım 3: Metrikleri hesapla
cd src
python evaluation.py
```

> ⚠️ **Önemli:** Spark ve Ollama aynı anda çalıştırılmaz — pipeline bunu otomatik yönetir.

---

## 🤖 Ajan Açıklamaları

### Ajan 1 — Analiz Ajanı
Ham hasta verisini inceler, referans aralık dışındaki tüm lab değerlerini tespit eder. Çıktı: JSON formatında anormal bulgular listesi.

### Ajan 2 — Karar Ajanı
Anormal bulguları SOFA/qSOFA kriterlerine göre değerlendirir ve 1–5 arası aciliyet skoru atar:
- `1` = Rutin takip
- `3` = 4-6 saat içinde müdahale
- `5` = Hayati tehlike, acil müdahale

### Ajan 3 — Raporlama Ajanı
İki ajanın çıktısını birleştirerek klinisyen dostu, maksimum 150 kelimelik paragraf formatında klinik özet üretir.

---

## 📈 Deneylerin Sonuçları

### Deney 1: Spark vs Pandas

| Yöntem | Süre | Satır |
|--------|------|-------|
| Pandas (CSV) | 1.25 sn | 107,727 |
| Spark (Pipeline) | 20.6 sn | 107,727+ |
| Parquet (Spark çıktısı) | 1.56 sn | 2,500 |

> Spark ilk çalıştırmada JVM başlatma maliyeti taşır; asıl avantaj GB boyutlu dosyalarda bellek kısıtı olmadan işlem yapabilmesidir.

### Deney 2: LLM Tutarlılık (3 tur)

| Metrik | Değer |
|--------|-------|
| Aciliyet skoru std dev | 0.0 (tam tutarlı) |
| Tüm turlarda aynı skor | ✅ Evet |

### Deney 3: Pipeline Sağlamlığı

| Metrik | Değer |
|--------|-------|
| Toplam vaka | 67 |
| A1 hata oranı | %0.0 |
| Zincirleme hata oranı | %0 |
| Genel hata oranı | %0.0 |

### Deney 4: Süre Analizi

| Metrik | Değer |
|--------|-------|
| Ortalama süre | 131.2 sn |
| Min süre | 34.0 sn |
| Max süre | 314.4 sn |
| Vaka/dakika | 0.5 |

---

## 🧪 Tespit Edilen Anormallikler

Pipeline şu 8 kritik lab testini izler:

| Test | Eşik | Klinik Anlam | Precision | Recall |
|------|------|--------------|-----------|--------|
| Lactate | > 2.0 mmol/L | Sepsis göstergesi | 1.0 | 0.865 |
| Creatinine | > 2.0 mg/dL | Böbrek yetmezliği | 1.0 | 0.963 |
| Hemoglobin | < 7.0 g/dL | Ciddi anemi | 1.0 | 0.700 |
| Potassium | > 5.5 mEq/L | Kardiyak risk | 1.0 | 0.808 |
| Sodium | < 130 mEq/L | Hiponatremi | 1.0 | 0.952 |
| Bicarbonate | < 18 mEq/L | Asidoz | 1.0 | 0.815 |
| BUN | > 40 mg/dL | Böbrek fonksiyonu | 1.0 | 0.929 |
| Chloride | < 95 mEq/L | Hipokloremi | 1.0 | 0.917 |

---

## ⚙️ Teknoloji Yığını

| Katman | Teknoloji | Amaç |
|--------|-----------|------|
| Büyük Veri | Apache Spark 3.x | GB boyutlu CSV işleme |
| Veri Formatı | Apache Parquet | Hızlı ara depolama |
| Ajan Çerçevesi | LangGraph | Durum makinesi / ajan orkestrasyonu |
| LLM | Ollama + Llama 3.2 3B | Yerel, ücretsiz çıkarım |
| LLM Entegrasyonu | LangChain Community | Prompt yönetimi |
| Veri Bilimi | Pandas, PyArrow | Analiz ve değerlendirme |
| Veri Seti | MIMIC-IV (PhysioNet) | Gerçek hasta verisi |

---

## ⚠️ Önemli Uyarı

> Bu sistem **yalnızca araştırma amaçlıdır.** Gerçek klinik karar vermede, hasta bakımında veya tıbbi tanı süreçlerinde kullanılmamalıdır. MIMIC-IV verisine erişim PhysioNet Veri Kullanım Sözleşmesi'ne tabidir.

---

## 📚 Atıflar

- Johnson, A. et al. (2023). MIMIC-IV. *PhysioNet.*
- LangChain AI. LangGraph: Agent Orchestration Framework.
- Apache Spark. Unified Analytics Engine for Large-Scale Data Processing.
- Meta AI. Llama 3.2: Lightweight Multilingual Models.

---

## 👤 Geliştirici
  
📧 [GitHub Profilim](https://github.com/esma6)

---

<div align="center">
  <sub>🔬 Tamamen açık kaynaklı · Yerel LLM · API ücreti yok · Tekrar edilebilir deneyler</sub>
</div>
