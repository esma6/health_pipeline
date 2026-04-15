# src/langgraph_pipeline.py
# Katman 2: LangGraph Durum Makinesi

import json
import time
from pathlib import Path
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from agents import analysis_agent, decision_agent, reporting_agent

RESULTS_DIR = Path(__file__).parent.parent / "results"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------
# DURUM TIPI
# ---------------------------------------------------

class PipelineState(TypedDict):
    patient_id: str
    raw_data: dict
    anomalies: Optional[dict]
    urgency: Optional[dict]
    report: Optional[str]
    error_log: list
    failed_agent: Optional[str]
    start_time: float
    duration_sec: Optional[float]


# ---------------------------------------------------
# DUGUM FONKSIYONLARI
# ---------------------------------------------------

def run_analysis(state: PipelineState) -> PipelineState:
    """Analiz Ajani'ni calistiran dugum."""
    print(f"  [A1] Analiz basliyor - Hasta: {state['patient_id']}")
    result = analysis_agent(state["raw_data"])

    if "error" in result:
        state["error_log"].append({"agent": "A1", "error": result["error"]})
        state["failed_agent"] = "A1"

    state["anomalies"] = result
    return state


def run_decision(state: PipelineState) -> PipelineState:
    """Karar Ajani'ni calistiran dugum."""
    print(f"  [A2] Karar analizi basliyor - Hasta: {state['patient_id']}")
    result = decision_agent(state["anomalies"])

    if "error" in result:
        state["error_log"].append({"agent": "A2", "error": result["error"]})
        if not state.get("failed_agent"):
            state["failed_agent"] = "A2"

    state["urgency"] = result
    return state


def run_reporting(state: PipelineState) -> PipelineState:
    """Raporlama Ajani'ni calistiran dugum."""
    print(f"  [A3] Rapor olusturuluyor - Hasta: {state['patient_id']}")
    result = reporting_agent(state["anomalies"], state["urgency"])

    state["report"] = result
    state["duration_sec"] = round(time.time() - state["start_time"], 1)
    print(f"  [Tamamlandi] Hasta {state['patient_id']} - {state['duration_sec']}sn")
    return state


def handle_error(state: PipelineState) -> PipelineState:
    """Hata yonetimi dugumu."""
    state["report"] = (
        f"[PIPELINE ERROR] Failed at: {state.get('failed_agent', 'Unknown')}. "
        f"Errors: {state['error_log']}"
    )
    state["duration_sec"] = round(time.time() - state["start_time"], 1)
    return state


# ---------------------------------------------------
# GRAF OLUSTURMA
# ---------------------------------------------------

def build_pipeline():
    """LangGraph durum makinesini insa eder ve derler."""
    graph = StateGraph(PipelineState)

    # Dugumleri ekle
    graph.add_node("analyze", run_analysis)
    graph.add_node("decide", run_decision)
    graph.add_node("report", run_reporting)
    graph.add_node("error_handler", handle_error)

    # Baslangic noktasi
    graph.set_entry_point("analyze")

    # Kosullu kenarlari tanimla
    graph.add_conditional_edges(
        "analyze",
        lambda s: "decide" if s["anomalies"] and "error" not in s["anomalies"] else "error_handler"
    )

    graph.add_conditional_edges(
        "decide",
        lambda s: "report" if s["urgency"] and "error" not in s["urgency"] else "error_handler"
    )

    graph.add_edge("report", END)
    graph.add_edge("error_handler", END)

    return graph.compile()


# ---------------------------------------------------
# COK HASTA ISLEME
# ---------------------------------------------------

def run_patient_pipeline(patient_records: list, max_patients: int = 50):
    """
    Birden fazla hasta icin pipeline'i calistirir.
    max_patients: Kacini isliyecegimiz (test icin 50, tam calisma icin 500+)
    """
    pipeline = build_pipeline()

    all_results = []
    error_count = 0

    print(f"\n{'=' * 60}")
    print(f"LANGGRAPH PIPELINE - {min(max_patients, len(patient_records))} hasta isleniyor")
    print(f"{'=' * 60}")

    for i, patient in enumerate(patient_records[:max_patients]):
        initial_state = {
            "patient_id": str(patient.get("subject_id", i)),
            "raw_data": patient,
            "anomalies": None,
            "urgency": None,
            "report": None,
            "error_log": [],
            "failed_agent": None,
            "start_time": time.time(),
            "duration_sec": None
        }

        result = pipeline.invoke(initial_state)
        all_results.append(result)

        if result.get("error_log"):
            error_count += 1

        # Her 10 hastada bir ilerleme goster
        if (i + 1) % 10 == 0:
            print(f"  [Ilerleme] {i + 1}/{min(max_patients, len(patient_records))} hasta tamamlandi")

    # Sonuclari kaydet
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "llm_results.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[Pipeline] Tamamlandi!")
    print(f"  - Toplam hasta: {len(all_results)}")
    print(f"  - Hata: {error_count} ({round(error_count / len(all_results) * 100, 1)}%)")
    print(f"  - Basarili: {len(all_results) - error_count}")
    print(f"  - Sonuclar: {output_path}")

    return all_results


# ---------------------------------------------------
# TEST - PARQUET'TEN HASTA OKU VE ISLE
# ---------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    print("Parquet dosyasi okunuyor...")
    df = pd.read_parquet(str(PROCESSED_DIR / "patient_anomalies.parquet"))
    print(f"Toplam kayit: {len(df):,}")

    # Her hasta icin TUM lab testlerini grupla
    # Onceki yaklasim: her hastadan sadece 1 satir -> modele tek test gidiyordu
    # Yeni yaklasim: her hastanin tum lab testleri tek sozlukte toplanir
    LAB_KOLONLAR = ["label", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag"]
    HASTA_KOLONLAR = ["subject_id", "hadm_id", "gender", "anchor_age", "hospital_expire_flag"]

    def safe_val(val):
        """Timestamp, NaN ve numpy tiplerini JSON-safe yap."""
        if hasattr(val, "isoformat"):
            return str(val)
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        import numpy as np
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        return val

    patient_records = []
    for subject_id, grup in df.groupby("subject_id"):
        hasta = {}
        for kol in HASTA_KOLONLAR:
            if kol in grup.columns:
                hasta[kol] = safe_val(grup.iloc[0][kol])

        lab_testleri = []
        for _, satir in grup.iterrows():
            test = {}
            for kol in LAB_KOLONLAR:
                if kol in grup.columns:
                    test[kol] = safe_val(satir[kol])
            lab_testleri.append(test)

        # Ref_range bilgisi olan ve valuenum dolu olan testleri sec
        # Kolon adi: parquet'ten "ref_range_lower"/"ref_range_upper" olarak geliyor
        # Maksimum 20 test gonder - token limitini asiyor
        lab_testleri = [
            t for t in lab_testleri
            if t.get("valuenum") is not None
            and t.get("ref_range_lower") is not None
            and t.get("ref_range_upper") is not None
        ][:20]

        hasta["lab_tests"] = lab_testleri
        hasta["test_count"] = len(lab_testleri)
        patient_records.append(hasta)

    print(f"Benzersiz hasta sayisi: {len(patient_records)}")
    ort = sum(p["test_count"] for p in patient_records) / len(patient_records)
    print(f"Hasta basina ortalama lab testi: {ort:.1f}")

    # Pipeline'i calistir
    print(f"\n{len(patient_records)} hasta ile calisma basliyor...\n")
    results = run_patient_pipeline(patient_records, max_patients=len(patient_records))

    print("\n--- Ornek Rapor (Hasta 1) ---")
    print(results[0].get("report", "Rapor yok"))