# src/agents.py
# Katman 2: LLM Ajan Tanimlari

import json
import time
from langchain_community.llms import Ollama

# Ollama baglantisi
MODEL_NAME = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"


def get_llm(temperature=0.1):
    """
    Ollama LLM nesnesi dondurur.
    temperature=0.1: Dusuk deger = daha tutarli, az yaratici cikti
    """
    return Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature
    )


# ---------------------------------------------------
# AJAN 1 - ANALIZ AJANI
# ---------------------------------------------------

ANALIZ_SYSTEM_PROMPT = """
You are a clinical data analyst. You analyze structured patient data.
Your task is to identify EVERY SINGLE abnormal value without exception.

CRITICAL RULES:
- You MUST check ALL lab tests present in the data, not just the most severe one
- Report EACH abnormal test as a SEPARATE entry in the list
- Do NOT skip or summarize any test - if 5 tests are abnormal, return 5 entries
- Check these tests explicitly: Lactate, Creatinine, Hemoglobin, Potassium, Sodium, Bicarbonate, BUN, Chloride

Normal reference ranges:
- Lactate: 0.5-2.0 mmol/L (abnormal if > 2.0)
- Creatinine: 0.6-1.2 mg/dL (abnormal if > 1.2)
- Hemoglobin: 12.0-17.5 g/dL (abnormal if < 12.0)
- Potassium: 3.5-5.0 mEq/L (abnormal if > 5.0 or < 3.5)
- Sodium: 136-145 mEq/L (abnormal if < 136 or > 145)
- Bicarbonate: 22-29 mEq/L (abnormal if < 22)
- BUN: 7-20 mg/dL (abnormal if > 20)
- Chloride: 98-106 mEq/L (abnormal if < 98)

For each abnormality provide:
- test_name: exact name of the lab test (e.g. "BUN", "Chloride")
- value: the patient's numeric value
- normal_range: the expected normal range as string
- severity: 'mild', 'moderate', or 'severe'

IMPORTANT: Respond ONLY with valid JSON. No explanations, no text outside JSON.
Format: {"abnormal_findings": [...ALL findings, one per test...], "total_count": number}
"""


# BUN label normalizasyonu - MIMIC-IV'te "Blood Urea Nitrogen" olarak geliyor
BUN_ALIASES = ["blood urea nitrogen", "urea nitrogen", "bun"]

def _normalize_label(label: str) -> str:
    """MIMIC-IV lab test isimlerini standart isimlere donusturur."""
    label_lower = str(label).lower().strip()
    if any(alias in label_lower for alias in BUN_ALIASES):
        return "BUN"
    return label  # diger testler zaten standart isimde


def _simplify_patient_data(patient_data: dict) -> dict:
    """
    Modele gonderilecek veriyi sadelestir:
    - Gereksiz kolonlari at (sadece label, valuenum, ref_range gonder)
    - BUN label'ini normalize et
    - None degerli satirlari atla
    """
    simplified = {
        "subject_id": patient_data.get("subject_id"),
        "gender": patient_data.get("gender"),
        "anchor_age": patient_data.get("anchor_age"),
        "lab_tests": []
    }

    for test in patient_data.get("lab_tests", []):
        label = test.get("label")
        valuenum = test.get("valuenum")
        if label is None or valuenum is None:
            continue
        simplified["lab_tests"].append({
            "test_name": _normalize_label(label),
            "value": valuenum,
            "unit": test.get("valueuom"),
            "ref_lower": test.get("ref_range_lower"),
            "ref_upper": test.get("ref_range_upper"),
        })

    return simplified


def analysis_agent(patient_data: dict) -> dict:
    """
    Ajan 1: Ham veriyi analiz eder, anormal degerleri tespit eder.
    - Veri oncelikle sadelestirilir (gereksiz kolonlar atilir)
    - BUN label normalizasyonu yapilir
    - JSON parse hatasinda 2 kez retry yapilir
    """
    llm = get_llm(temperature=0.1)

    # Veriyi sadelestir ve normalize et
    simplified = _simplify_patient_data(patient_data)
    data_str = json.dumps(simplified, indent=2)

    prompt = f"""Analyze the following patient lab results and identify all abnormal values.

Patient Data:
{data_str}

Return ONLY a JSON object with abnormal findings."""

    def parse_response(response: str):
        clean = response.strip()
        for prefix in ["```json", "```"]:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip())

    # 3 deneme yap
    last_error = None
    for attempt in range(3):
        try:
            response = llm.invoke(prompt, system=ANALIZ_SYSTEM_PROMPT)
            return parse_response(response)
        except json.JSONDecodeError as e:
            last_error = f"JSON parse failed (attempt {attempt+1}): {str(e)}"
            if attempt < 2:
                time.sleep(1)
        except Exception as e:
            return {"error": f"Agent failed: {str(e)}"}

    return {"error": last_error}


# ---------------------------------------------------
# AJAN 2 - KARAR AJANI
# ---------------------------------------------------

KARAR_SYSTEM_PROMPT = """
You are a clinical decision support system.
You receive a list of abnormal clinical findings.

Assign an urgency score from 1 to 5 using STRICT criteria:

Score 1 - Routine monitoring:
  - Single mild electrolyte deviation (e.g. Chloride 94, Sodium 134)
  - No symptoms, stable patient

Score 2 - Follow-up within 24 hours:
  - Mild anemia (Hemoglobin 10-12)
  - Mild renal impairment (Creatinine 1.3-2.0)
  - Mild electrolyte imbalance

Score 3 - Follow-up within 4-6 hours:
  - Moderate anemia (Hemoglobin 7-10)
  - Moderate renal failure (Creatinine 2.0-3.0)
  - BUN > 40 without other organ involvement
  - Bicarbonate < 18

Score 4 - Urgent intervention needed:
  - Severe anemia (Hemoglobin < 7)
  - Severe renal failure (Creatinine > 3.0)
  - Lactate 2.0-4.0 (tissue hypoperfusion)
  - Multiple simultaneous abnormalities (3+)

Score 5 - Immediate life-threatening:
  - Lactate > 4.0 (severe shock)
  - Potassium > 6.5 (lethal hyperkalemia)
  - Sodium < 120 (severe hyponatremia with seizure risk)
  - 4+ simultaneous critical abnormalities

CALIBRATION RULE: Most hospitalized patients with lab anomalies score 2-3.
Score 4-5 should be assigned only when there is clear evidence of organ failure
or hemodynamic compromise. Do NOT default to 4 for all patients.

IMPORTANT: Respond ONLY with valid JSON.
Format: {"urgency_score": 1-5, "primary_concern": "text", "top_risks": ["risk1", "risk2", "risk3"]}
"""


def decision_agent(anomaly_data: dict) -> dict:
    """
    Ajan 2: Anormal bulgulari alir, aciliyet skoru atar.
    anomaly_data: Analiz ajanindan gelen anormal bulgular
    Returns: Aciliyet skoru ve risk degerlendirmesi
    """
    # Onceki ajan hata verdiyse bunu yakala ve ilet
    if "error" in anomaly_data:
        return {"error": "Upstream agent failed", "cascading_error": True}

    llm = get_llm(temperature=0.1)

    data_str = json.dumps(anomaly_data, indent=2)

    prompt = f"""Based on these clinical findings, assign an urgency score.

Findings:
{data_str}

Return ONLY a JSON urgency assessment."""

    try:
        response = llm.invoke(prompt, system=KARAR_SYSTEM_PROMPT)
        clean = response.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip())

    except Exception as e:
        return {"error": str(e), "cascading_error": True}


# ---------------------------------------------------
# AJAN 3 - RAPORLAMA AJANI
# ---------------------------------------------------

RAPOR_SYSTEM_PROMPT = """
You are a clinical report writer.
You receive structured clinical findings and urgency scores.

Write a concise clinical summary that:
1. Is maximum 150 words
2. Uses plain medical language
3. Highlights the most urgent findings FIRST
4. Suggests immediate actions if urgency score >= 4
5. Is written in paragraph form, NOT bullet points

Write the report in English. Be direct and clinical.
"""


def reporting_agent(anomaly_data: dict, urgency_data: dict) -> str:
    """
    Ajan 3: Her iki ajanin ciktisini klinisyen dostu rapora donusturur.
    Returns: Metin formatinda klinik rapor
    """
    # Yukaridan gelen hatalar varsa raporla
    if "error" in urgency_data:
        return f"[REPORT ERROR] Pipeline error: {urgency_data['error']}"

    llm = get_llm(temperature=0.3)  # Biraz daha yaratici olabilir

    combined = {
        "findings": anomaly_data,
        "urgency_assessment": urgency_data
    }

    prompt = f"""Write a clinical summary for the following assessment.

Clinical Data:
{json.dumps(combined, indent=2)}

Write a concise paragraph report (max 150 words)."""

    try:
        return llm.invoke(prompt, system=RAPOR_SYSTEM_PROMPT)
    except Exception as e:
        return f"[REPORT GENERATION FAILED]: {str(e)}"


# ---------------------------------------------------
# TEST BLOGU
# ---------------------------------------------------

if __name__ == "__main__":

    test_patient = {
        "subject_id": 12345,
        "hadm_id": 99999,
        "lab_results": [
            {"test": "Lactate", "value": 4.2, "unit": "mmol/L"},
            {"test": "Creatinine", "value": 3.1, "unit": "mg/dL"},
            {"test": "Hemoglobin", "value": 5.8, "unit": "g/dL"},
        ]
    }

    print("--- Ajan 1 Test ---")
    ajan1_cikti = analysis_agent(test_patient)
    print(json.dumps(ajan1_cikti, indent=2))

    print("\n--- Ajan 2 Test ---")
    ajan2_cikti = decision_agent(ajan1_cikti)
    print(json.dumps(ajan2_cikti, indent=2))

    print("\n--- Ajan 3 Test ---")
    rapor = reporting_agent(ajan1_cikti, ajan2_cikti)
    print(rapor)