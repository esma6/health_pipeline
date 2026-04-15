# src/evaluation.py
# Geliştirilmiş metrik hesaplama - ground truth dahil

import time
import json
import statistics
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ---------------------------------------------------
# DENEY 1: SPARK VS PANDAS - ADIL KARSILASTIRMA
# ---------------------------------------------------

def benchmark_pandas_vs_spark():
    """
    Adil karsilastirma:
    - Pandas: ham CSV'den okuma + filtreleme (Spark ile ayni is)
    - Spark suresi: spark_benchmark.json'dan alinir
    - Parquet: Spark ciktisini pandas ile okuma hizi
    """
    results = {}
    anomaly_ids = [50813, 50882, 50912, 51222, 50971, 50983, 51006, 50902]
    csv_path = str(RAW_DIR / "hosp" / "labevents.csv")

    # Pandas: CSV okuma + filtreleme
    print("Pandas CSV okuma + filtreleme testi...")
    t0 = time.time()
    df_csv = pd.read_csv(csv_path)
    df_filtered = df_csv[df_csv["itemid"].isin(anomaly_ids) & df_csv["valuenum"].notna()]
    pandas_duration = round(time.time() - t0, 2)
    pandas_rows = len(df_csv)
    filtered_rows = len(df_filtered)
    del df_csv, df_filtered
    print(f"  Pandas: {pandas_duration}sn, {pandas_rows:,} satir okundu, {filtered_rows:,} anormali")
    results["pandas_csv_duration_sec"] = pandas_duration
    results["pandas_total_rows"] = pandas_rows
    results["pandas_filtered_rows"] = filtered_rows

    # Parquet okuma
    print("Parquet okuma testi (Spark ciktisi)...")
    t0 = time.time()
    df_parquet = pd.read_parquet(str(PROCESSED_DIR / "lab_anomalies.parquet"))
    parquet_duration = round(time.time() - t0, 2)
    parquet_rows = len(df_parquet)
    del df_parquet
    print(f"  Parquet: {parquet_duration}sn, {parquet_rows:,} satir")
    results["parquet_read_duration_sec"] = parquet_duration
    results["parquet_rows"] = parquet_rows
    results["parquet_vs_csv_speedup"] = round(pandas_duration / parquet_duration, 2)

    # Spark suresi
    spark_bench_path = RESULTS_DIR / "spark_benchmark.json"
    if spark_bench_path.exists():
        with open(spark_bench_path) as f:
            spark_bench = json.load(f)
        spark_duration = spark_bench.get("lab_processing", {}).get("duration_sec")
        results["spark_pipeline_duration_sec"] = spark_duration
        if spark_duration:
            results["spark_vs_pandas_speedup"] = round(pandas_duration / spark_duration, 2)
            print(f"  Spark pipeline: {spark_duration}sn")
            print(f"  Spark vs Pandas hiz orani: {results['spark_vs_pandas_speedup']}x")
    else:
        print("  UYARI: spark_benchmark.json bulunamadi")

    print(f"  Parquet vs CSV hiz orani: {results['parquet_vs_csv_speedup']}x")
    return results


# ---------------------------------------------------
# DENEY 2: TUTARLILIK METRIKLERI
# ---------------------------------------------------

def calculate_consistency(patient_data: dict, runs: int = 3):
    """Ayni hasta verisini runs kez isleterek tutarlilik olcumu yapar."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from agents import analysis_agent, decision_agent

    urgency_scores = []
    all_findings = []

    for i in range(runs):
        print(f"  Calistirma {i + 1}/{runs}...")
        anomalies = analysis_agent(patient_data)
        if "error" not in anomalies:
            urgency = decision_agent(anomalies)
            if "urgency_score" in urgency:
                urgency_scores.append(urgency["urgency_score"])
            findings = set(
                f.get("test_name", "") for f in anomalies.get("abnormal_findings", [])
            )
            all_findings.append(findings)

    score_consistency = {
        "scores": urgency_scores,
        "mean": round(statistics.mean(urgency_scores), 2) if urgency_scores else None,
        "std_dev": round(statistics.stdev(urgency_scores), 2) if len(urgency_scores) > 1 else 0,
        "is_consistent": statistics.stdev(urgency_scores) <= 0.5 if len(urgency_scores) > 1 else True
    }

    jaccard_scores = []
    for i in range(len(all_findings)):
        for j in range(i + 1, len(all_findings)):
            if all_findings[i] or all_findings[j]:
                inter = len(all_findings[i] & all_findings[j])
                union = len(all_findings[i] | all_findings[j])
                jaccard_scores.append(inter / union if union > 0 else 0)

    finding_consistency = {
        "jaccard_mean": round(statistics.mean(jaccard_scores), 3) if jaccard_scores else None,
        "is_consistent": statistics.mean(jaccard_scores) >= 0.8 if jaccard_scores else False
    }

    return {"score_consistency": score_consistency, "finding_consistency": finding_consistency}


# ---------------------------------------------------
# DENEY 3: ZINCIRLEME HATA (PIPELINE SAGLAMLIGI)
# ---------------------------------------------------

def analyze_cascading_errors(results: list):
    """JSON/baglanti hatasi orani - klinik dogruluk degil, pipeline saglamligi."""
    total = len(results)
    a1_errors = sum(1 for r in results if r.get("failed_agent") == "A1")
    a2_errors = sum(1 for r in results if r.get("failed_agent") == "A2")
    any_error = sum(1 for r in results if r.get("error_log"))
    a1_fail_cases = [r for r in results if r.get("failed_agent") == "A1"]
    a2_after_a1 = sum(
        1 for r in a1_fail_cases
        if r.get("urgency") and "error" in r.get("urgency", {})
    )
    return {
        "total_cases": total,
        "a1_error_count": a1_errors,
        "a1_error_rate_pct": round(a1_errors / total * 100, 1),
        "a2_error_count": a2_errors,
        "a2_error_rate_pct": round(a2_errors / total * 100, 1),
        "cascade_rate_pct": round(a2_after_a1 / len(a1_fail_cases) * 100, 1) if a1_fail_cases else 0,
        "overall_error_rate_pct": round(any_error / total * 100, 1),
        "note": "Pipeline saglamligi olcusu (JSON/baglanti hatasi) - klinik dogruluk degil"
    }


# ---------------------------------------------------
# DENEY 4: SURE ANALIZI
# ---------------------------------------------------

def analyze_timing(results: list):
    durations = [r.get("duration_sec") for r in results if r.get("duration_sec")]
    if not durations:
        return {"error": "duration_sec verisi bulunamadi"}
    return {
        "mean_sec": round(statistics.mean(durations), 1),
        "min_sec": round(min(durations), 1),
        "max_sec": round(max(durations), 1),
        "std_dev_sec": round(statistics.stdev(durations), 1) if len(durations) > 1 else 0,
        "cases_per_minute": round(60 / statistics.mean(durations), 1),
        "total_cases": len(durations)
    }


# ---------------------------------------------------
# GROUND TRUTH 1: URGENCY vs MORTALITE KORELASYONU
# ---------------------------------------------------

def evaluate_urgency_vs_mortality(llm_results: list):
    """
    Ajanin urgency skoru ile hospital_expire_flag korelasyonu.
    Hipotez: urgency skoru yuksek hastalar daha fazla hayatini kaybetmis olmali.
    """
    df_patients = pd.read_parquet(str(PROCESSED_DIR / "patient_anomalies.parquet"))

    records = []
    for r in llm_results:
        pid = str(r.get("patient_id"))
        urgency_data = r.get("urgency", {})
        urgency_score = urgency_data.get("urgency_score") if isinstance(urgency_data, dict) else None
        if urgency_score is None:
            continue
        patient_row = df_patients[df_patients["subject_id"].astype(str) == pid]
        if patient_row.empty:
            continue
        expire_flag = int(patient_row.iloc[0].get("hospital_expire_flag", 0))
        records.append({
            "subject_id": pid,
            "urgency_score": urgency_score,
            "hospital_expire_flag": expire_flag
        })

    if not records:
        return {"error": "Eslestirilecek hasta bulunamadi"}

    df = pd.DataFrame(records)
    died = df[df["hospital_expire_flag"] == 1]
    survived = df[df["hospital_expire_flag"] == 0]

    result = {
        "total_matched": len(df),
        "died_count": len(died),
        "survived_count": len(survived),
        "mortality_rate_pct": round(len(died) / len(df) * 100, 1),
        "mean_urgency_died": round(died["urgency_score"].mean(), 2) if len(died) > 0 else None,
        "mean_urgency_survived": round(survived["urgency_score"].mean(), 2) if len(survived) > 0 else None,
    }

    high = df[df["urgency_score"] >= 4]
    low = df[df["urgency_score"] < 4]
    if len(high) > 0:
        result["mortality_high_urgency_pct"] = round(high["hospital_expire_flag"].mean() * 100, 1)
        result["high_urgency_count"] = len(high)
    if len(low) > 0:
        result["mortality_low_urgency_pct"] = round(low["hospital_expire_flag"].mean() * 100, 1)
        result["low_urgency_count"] = len(low)

    corr = df["urgency_score"].corr(df["hospital_expire_flag"])
    result["pearson_correlation"] = round(corr, 3)
    result["interpretation"] = (
        "Pozitif korelasyon: yuksek urgency = daha fazla olum (beklenen)" if corr > 0.1
        else "Zayif korelasyon: model olumu onceden goremedı" if corr >= 0
        else "Negatif korelasyon: beklenmedik sonuc, model kalibrasyonu gerekli"
    )
    return result


# ---------------------------------------------------
# GROUND TRUTH 2: REF_RANGE ILE PRECISION / RECALL
# ---------------------------------------------------

def evaluate_precision_recall_vs_ref_range(llm_results: list):
    """
    MIMIC-IV'un kendi ref_range_lower/upper degerlerini ground truth olarak kullanir.
    
    Ground truth (gercekten anormal):
        valuenum < ref_range_lower VEYA valuenum > ref_range_upper
    
    Model tahmini:
        Ajanin abnormal_findings listesinde o test var mi?
    
    Bu yaklasim ICD'ye gore cok daha saglamdir cunku:
    - Dogrudan olcum verisine dayanir
    - Klinisyen yorumuna ihtiyac duymaz
    - Her test icin ayri esik degeri kullanir
    """

    # MIMIC-IV lab test adi -> parquet'teki label eslestirmesi
    # (model bu isimlerle donduruyor, parquet'te label kolonu var)
    TEST_NAME_MAP = {
        "Lactate": ["lactate", "lactic acid"],
        "Creatinine": ["creatinine"],
        "Hemoglobin": ["hemoglobin"],
        "Potassium": ["potassium"],
        "Sodium": ["sodium"],
        "Bicarbonate": ["bicarbonate"],
        "BUN": ["urea nitrogen", "bun"],
        "Chloride": ["chloride"],
    }

    # Parquet'ten ref_range bilgisi olan satirlari oku
    df = pd.read_parquet(str(PROCESSED_DIR / "patient_anomalies.parquet"))

    # ref_range kolonlari numerik yap
    df["ref_range_lower"] = pd.to_numeric(df["ref_range_lower"], errors="coerce")
    df["ref_range_upper"] = pd.to_numeric(df["ref_range_upper"], errors="coerce")
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")

    # Ground truth: ref_range bilgisi olan ve gercekten anormal olan satirlar
    df_with_range = df[
        df["ref_range_lower"].notna() &
        df["ref_range_upper"].notna() &
        df["valuenum"].notna()
    ].copy()

    df_with_range["truly_abnormal"] = (
        (df_with_range["valuenum"] < df_with_range["ref_range_lower"]) |
        (df_with_range["valuenum"] > df_with_range["ref_range_upper"])
    )

    # Her hasta icin gercekten anormal olan test isimlerini bul
    def get_truly_abnormal_tests(pid):
        patient_rows = df_with_range[df_with_range["subject_id"].astype(str) == str(pid)]
        abnormal = patient_rows[patient_rows["truly_abnormal"] == True]
        tests = set()
        for label in abnormal["label"].dropna():
            label_lower = str(label).lower()
            for canonical, aliases in TEST_NAME_MAP.items():
                if any(alias in label_lower for alias in aliases):
                    tests.add(canonical)
        return tests

    tp_total = fp_total = fn_total = evaluated = 0
    per_test_stats = {name: {"tp": 0, "fp": 0, "fn": 0} for name in TEST_NAME_MAP}

    for r in llm_results:
        pid = str(r.get("patient_id"))
        anomalies = r.get("anomalies", {})
        if not isinstance(anomalies, dict):
            continue

        # Model tahmini
        detected = set(
            f.get("test_name", "").strip()
            for f in anomalies.get("abnormal_findings", [])
        )

        # Gercek ground truth
        truly_abnormal = get_truly_abnormal_tests(pid)

        if not truly_abnormal and not detected:
            continue  # Her ikisi de bos, atla

        evaluated += 1

        tp = len(detected & truly_abnormal)
        fp = len(detected - truly_abnormal)
        fn = len(truly_abnormal - detected)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        # Test bazinda istatistik
        for test in TEST_NAME_MAP:
            if test in detected and test in truly_abnormal:
                per_test_stats[test]["tp"] += 1
            elif test in detected and test not in truly_abnormal:
                per_test_stats[test]["fp"] += 1
            elif test not in detected and test in truly_abnormal:
                per_test_stats[test]["fn"] += 1

    if evaluated == 0:
        return {"error": "Hicbir hasta icin ref_range verisi bulunamadi"}

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Test bazinda precision/recall
    per_test_results = {}
    for test, counts in per_test_stats.items():
        t, f_p, f_n = counts["tp"], counts["fp"], counts["fn"]
        p = t / (t + f_p) if (t + f_p) > 0 else None
        r = t / (t + f_n) if (t + f_n) > 0 else None
        if p is not None or r is not None:
            per_test_results[test] = {
                "precision": round(p, 3) if p is not None else None,
                "recall":    round(r, 3) if r is not None else None,
                "tp": t, "fp": f_p, "fn": f_n
            }

    return {
        "ground_truth_method": "MIMIC-IV ref_range_lower/upper (dogrudan olcum)",
        "evaluated_patients": evaluated,
        "true_positives":  tp_total,
        "false_positives": fp_total,
        "false_negatives": fn_total,
        "overall_precision": round(precision, 3),
        "overall_recall":    round(recall, 3),
        "overall_f1":        round(f1, 3),
        "per_test_breakdown": per_test_results,
        "methodology_note": (
            "Ground truth: valuenum < ref_range_lower veya > ref_range_upper. "
            "Model tahmini: abnormal_findings listesindeki test isimleri. "
            "ICD proxy'sine gore cok daha dogrudan bir karsilastirmadir."
        )
    }


# ---------------------------------------------------
# ANA CALISTIRICI
# ---------------------------------------------------

if __name__ == "__main__":

    all_metrics = {}

    print("\n" + "=" * 60)
    print("DENEY 1: Spark vs Pandas (Adil Karsilastirma)")
    print("=" * 60)
    deney1 = benchmark_pandas_vs_spark()
    print(json.dumps(deney1, indent=2))
    all_metrics["deney1_spark_vs_pandas"] = deney1

    print("\n" + "=" * 60)
    print("DENEY 2: LLM Tutarlilik Testi (3 tur)")
    print("=" * 60)
    test_case = {
        "subject_id": 12345,
        "lab_results": [
            {"test": "Lactate", "value": 4.2},
            {"test": "Creatinine", "value": 3.8},
        ]
    }
    deney2 = calculate_consistency(test_case, runs=3)
    print(json.dumps(deney2, indent=2))
    all_metrics["deney2_tutarlilik"] = deney2

    results_path = RESULTS_DIR / "llm_results.json"
    if not results_path.exists():
        print("\nUYARI: llm_results.json bulunamadi! Once langgraph_pipeline.py'yi calistirin.")
    else:
        with open(results_path) as f:
            llm_results = json.load(f)

        print("\n" + "=" * 60)
        print("DENEY 3: Pipeline Saglamligi (Zincirleme Hata)")
        print("=" * 60)
        deney3 = analyze_cascading_errors(llm_results)
        print(json.dumps(deney3, indent=2))
        all_metrics["deney3_pipeline_saglamligi"] = deney3

        print("\n" + "=" * 60)
        print("DENEY 4: Sure Analizi")
        print("=" * 60)
        deney4 = analyze_timing(llm_results)
        print(json.dumps(deney4, indent=2))
        all_metrics["deney4_sure"] = deney4

        print("\n" + "=" * 60)
        print("GROUND TRUTH 1: Urgency vs Mortalite Korelasyonu")
        print("=" * 60)
        gt1 = evaluate_urgency_vs_mortality(llm_results)
        print(json.dumps(gt1, indent=2))
        all_metrics["ground_truth_1_mortalite"] = gt1

        print("\n" + "=" * 60)
        print("GROUND TRUTH 2: ref_range ile Precision/Recall")
        print("=" * 60)
        gt2 = evaluate_precision_recall_vs_ref_range(llm_results)
        print(json.dumps(gt2, indent=2))
        all_metrics["ground_truth_2_ref_range"] = gt2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Evaluation] Tum metrikler kaydedildi: results/all_metrics.json")