"""DreamCatcher web service.

Run locally (inside dreamcatcher_app):   python -m uvicorn app.server:app --reload --port 8000
Run with Docker:                         docker build -t dreamcatcher_img .   then   docker run --name dreamcatcher -p 8000:8000 dreamcatcher_img
Pages:  /  (API check)   /docs  (Swagger UI)   /web/  (the one-page application)
"""
import json
from pathlib import Path
import re
from typing import List, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.dreamcatcher_utils import OPTIONS, TIE_ORDER, apply_clinical_rule, build_features, decide, tied_classes

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / "models" / "dreamcatcher_model.joblib")       # trained pipeline (preprocessing + SMOTE + SelectKBest + KNN)
encoder = joblib.load(BASE / "models" / "label_encoder.pkl")             # 0/1/2 -> class names
card = json.loads((BASE / "models" / "model_card.json").read_text())     # metrics, options, screening thresholds

app = FastAPI(title="DreamCatcher API", version="1.0",
              description="Sleep-disorder screening for university students. Educational aid, not a diagnosis.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

EXAMPLE_PERSON = {"gender": "Female", "age": 22, "sleep_duration": "6-7 hours", "sleep_quality": "Poor", "stress_level": "Moderate",
                  "physical_activity": 30, "bmi_category": "Normal", "blood_pressure": "118/76", "heart_rate": 82, "daily_steps": "5,000-7,999"}
EXAMPLE_SHORT_SLEEPER = {"gender": "Male", "age": 21, "sleep_duration": "Less than 4 hours", "sleep_quality": "Very Poor", "stress_level": "High",
                         "physical_activity": 20, "bmi_category": "Overweight", "blood_pressure": "130/85", "heart_rate": 88, "daily_steps": "Less than 4,999"}
EXAMPLE_GOOD_SLEEPER = {"gender": "Female", "age": 22, "sleep_duration": "8 hours or more", "sleep_quality": "Good", "stress_level": "Low",
                        "physical_activity": 60, "bmi_category": "Normal", "blood_pressure": "120/80", "heart_rate": 72, "daily_steps": "8,000-11,999"}


class SleepInput(BaseModel):
    model_config = {"json_schema_extra": {"examples": [EXAMPLE_PERSON]}}      # pre-fills "Try it out" in Swagger

    gender: Literal["Male", "Female"]
    age: int = Field(ge=16, le=60, description="age in years")
    department: str = Field("Other", description="optional; not used by the deployed model")
    academic_level: Optional[Literal["Level-1", "Level-2", "Level-3", "Level-4"]] = Field(None, description="optional; not used by the deployed model")
    sleep_duration: Literal["Less than 4 hours", "4-5 hours", "6-7 hours", "8 hours or more"]
    sleep_quality: Literal["Very Poor", "Poor", "Average", "Good", "Excellent"]
    stress_level: Literal["Very Low", "Low", "Moderate", "High", "Very High"]
    physical_activity: int = Field(ge=0, le=1440, description="minutes of physical activity per day (a day has 1,440 minutes; the model treats values above 180 as 180)")
    bmi_category: Literal["Underweight", "Normal", "Overweight", "Obese"]
    blood_pressure: str = Field(pattern=r"^\d{2,3}/\d{2,3}$", examples=["120/80"])
    heart_rate: int = Field(ge=40, le=200, description="beats per minute")
    daily_steps: Literal["Less than 4,999", "5,000-7,999", "8,000-11,999", "12,000-15,999", "More than 16,000"]

    @field_validator("blood_pressure")
    @classmethod
    def check_blood_pressure(cls, v: str) -> str:
        sys_, dia = (int(x) for x in re.match(r"^(\d{2,3})/(\d{2,3})$", v).groups())
        if not 70 <= sys_ <= 220:
            raise ValueError("systolic pressure must be between 70 and 220 mmHg")
        if not 40 <= dia <= 130:
            raise ValueError("diastolic pressure must be between 40 and 130 mmHg")
        if sys_ <= dia:
            raise ValueError("systolic pressure must be higher than diastolic pressure")
        return v


class BatchInput(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"people": [EXAMPLE_SHORT_SLEEPER, EXAMPLE_PERSON, EXAMPLE_GOOD_SLEEPER]}]}}
    people: List[SleepInput] = Field(min_length=1, max_length=20)


# ----------------------------------------------------------------------------- numeric-array input
# Order and codes of the 11 values in each row of POST /predict-features (codes follow the order of the answers).
FEATURE_ORDER = ["gender", "age", "sleep_duration", "sleep_quality", "stress_level", "physical_activity",
                 "bmi_category", "systolic", "diastolic", "heart_rate", "daily_steps"]
GENDER_CODES = ["Female", "Male"]                      # 0 = Female, 1 = Male
CODED = {"gender": GENDER_CODES, "sleep_duration": OPTIONS["sleep_duration"], "sleep_quality": OPTIONS["sleep_quality"],
         "stress_level": OPTIONS["stress_level"], "bmi_category": OPTIONS["bmi_category"], "daily_steps": OPTIONS["daily_steps"]}


class FeaturesInput(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"features": [[1, 21, 0, 0, 3, 20, 2, 130, 85, 88, 0],
                                                                      [0, 22, 2, 1, 2, 30, 1, 118, 76, 82, 1],
                                                                      [0, 22, 3, 3, 1, 60, 1, 120, 80, 72, 2]]}]}}
    features: List[List[int]] = Field(min_length=1, max_length=20,
                                      description="one row of 11 numbers per person, in the order listed in the endpoint description")


def row_to_input(row: List[int], n: int) -> SleepInput:
    """[0, 22, 2, 1, 3, 40, 1, 115, 75, 78, 1] -> SleepInput (readable answers), with a clear message for a bad row."""
    if len(row) != len(FEATURE_ORDER):
        raise HTTPException(422, f"row {n}: expected {len(FEATURE_ORDER)} values ({', '.join(FEATURE_ORDER)}), got {len(row)}")
    v = dict(zip(FEATURE_ORDER, row))
    for k, labels in CODED.items():
        if not 0 <= v[k] < len(labels):
            raise HTTPException(422, f"row {n}: {k} must be a code from 0 to {len(labels) - 1}, got {v[k]}")
        v[k] = labels[v[k]]
    v["blood_pressure"] = f"{v.pop('systolic')}/{v.pop('diastolic')}"
    try:
        return SleepInput(**v)
    except ValidationError as e:
        first = e.errors()[0]
        raise HTTPException(422, f"row {n}: {first['loc'][-1] if first['loc'] else 'value'}: {first['msg'].replace('Value error, ', '')}")


# ----------------------------------------------------------------------------- endpoints
@app.get("/")
def root():
    return {"message": "DreamCatcher API"}


@app.get("/health")
def health():
    return {"status": "ok", "model": card["model"], "trained_on": card["trained_on"]}


@app.get("/options")
def options():
    """Allowed answers for every form field (the web page builds its dropdowns from this)."""
    return card["options"]


@app.get("/model-card")
def model_card():
    return card


def advice(any_p: float, th: dict) -> str:
    if any_p >= th.get("balanced_youden", 0.5):
        return "Higher likelihood"
    if any_p >= th.get("high_sensitivity_90", 0.5):
        return "Possible"
    return "Lower likelihood"


def score(p: SleepInput) -> dict:
    X = build_features(pd.DataFrame([p.model_dump()]))
    classes = list(encoder.classes_)
    raw = model.predict_proba(X)
    proba, ruled = apply_clinical_rule(raw, X, classes)      # Insomnia ruled out for 8+ hours with good/excellent quality
    proba, ruled = proba[0], bool(ruled[0])
    i = int(decide([proba], classes)[0])                     # highest probability; ties go to a disorder (TIE_ORDER)
    probs = {c: round(float(v), 4) for c, v in zip(classes, proba)}
    any_p = round(1 - probs["No Sleep Disorder"], 4)
    th = card.get("screening_thresholds", {})
    return {
        "predicted_class": classes[i],
        "confidence": round(float(proba[i]), 2),
        "screening_advice": advice(any_p, th),
        "tied_with": [c for c in tied_classes(proba, classes) if c != classes[i]],
        "ranking": sorted(classes, key=lambda c: (-probs[c], TIE_ORDER.index(c))),
        "probabilities": probs,
        "model_probabilities": {c: round(float(v), 4) for c, v in zip(classes, raw[0])},
        "clinical_rule_applied": ruled,
        "any_disorder_probability": any_p,
        "screening_flag": any_p >= th.get("balanced_youden", 0.5),
        "high_sensitivity_flag": any_p >= th.get("high_sensitivity_90", 0.5),
        "thresholds": th,
        "disclaimer": card["disclaimer"],
    }


@app.post("/predict")
def predict(p: SleepInput):
    """One person, answers written out (the example is pre-filled under "Try it out")."""
    return score(p)


@app.post("/predict-batch")
def predict_batch(b: BatchInput):
    """Several people at once (up to 20); returns {"results": [...]} in the same order. Used by the web page."""
    return {"results": [score(p) for p in b.people]}


@app.post("/predict-features")
def predict_features(f: FeaturesInput):
    """Numbers only: one row of 11 values per person, in this order:

    1. gender (0 = Female, 1 = Male)
    2. age in years (16-60)
    3. sleep_duration (0 = Less than 4 hours, 1 = 4-5 hours, 2 = 6-7 hours, 3 = 8 hours or more)
    4. sleep_quality (0 = Very Poor, 1 = Poor, 2 = Average, 3 = Good, 4 = Excellent)
    5. stress_level (0 = Very Low, 1 = Low, 2 = Moderate, 3 = High, 4 = Very High)
    6. physical_activity in minutes per day (0-1440)
    7. bmi_category (0 = Underweight, 1 = Normal, 2 = Overweight, 3 = Obese)
    8. systolic blood pressure in mmHg (70-220)
    9. diastolic blood pressure in mmHg (40-130, lower than systolic)
    10. heart_rate in beats per minute (40-200)
    11. daily_steps (0 = Less than 4,999, 1 = 5,000-7,999, 2 = 8,000-11,999, 3 = 12,000-15,999, 4 = More than 16,000)

    Returns {"results": [{"predicted_class", "confidence", "screening_advice", "any_disorder_probability"}, ...]} in the same order.
    """
    out = []
    for n, row in enumerate(f.features, start=1):
        r = score(row_to_input(row, n))
        out.append({k: r[k] for k in ("predicted_class", "confidence", "screening_advice", "any_disorder_probability")})
    return {"results": out}


app.mount("/web", StaticFiles(directory=BASE / "static", html=True), name="web")   # keep last
