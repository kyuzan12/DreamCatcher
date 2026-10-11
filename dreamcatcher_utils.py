"""Shared cleaning + feature engineering for DreamCatcher (used by the notebook AND the FastAPI app)."""
import re
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.metrics import f1_score

RENAME = {
    "Department": "department", "Gender": "gender", "Age": "age", "Sleep Duration": "sleep_duration",
    "Quality of Sleep": "sleep_quality", "Physical Activity Level": "physical_activity",
    "Stress Level": "stress_level", "BMI Category": "bmi_category",
    "Blood Pressure (systolic/diastolic)": "blood_pressure", "Heart Rate (bpm)": "heart_rate",
    "Daily Steps": "daily_steps", "Sleep Disorder": "sleep_disorder", "Academic Level": "academic_level",
}
CLASSES = ["Insomnia", "No Sleep Disorder", "Sleep Apnea"]

# Ordered option lists: index in the list == ordinal code (academic_level starts at 1)
OPTIONS = {
    "gender": ["Male", "Female"],
    "department": ["CSE", "Mathematics", "ECE", "Agriculture", "Other"],
    "academic_level": ["Level-1", "Level-2", "Level-3", "Level-4"],
    "sleep_duration": ["Less than 4 hours", "4-5 hours", "6-7 hours", "8 hours or more"],
    "sleep_quality": ["Very Poor", "Poor", "Average", "Good", "Excellent"],
    "stress_level": ["Very Low", "Low", "Moderate", "High", "Very High"],
    "bmi_category": ["Underweight", "Normal", "Overweight", "Obese"],
    "daily_steps": ["Less than 4,999", "5,000-7,999", "8,000-11,999", "12,000-15,999", "More than 16,000"],
}
NUM_COLS = ["age", "physical_activity", "heart_rate", "systolic", "diastolic", "map", "pulse_pressure"]
ORD_COLS = ["sleep_duration", "sleep_quality", "stress_level", "bmi_category", "daily_steps", "academic_level"]
CAT_COLS = ["gender", "department"]
FEATURES = NUM_COLS + ORD_COLS + CAT_COLS


def norm_text(x):
    """Lower-case, unify dashes, collapse whitespace."""
    return re.sub(r"\s+", " ", str(x).replace("\u2013", "-").replace("\u2014", "-")).strip().lower()


def _ordinal(options):
    lookup = {norm_text(o): i for i, o in enumerate(options)}
    def f(x):
        s = norm_text(x)
        for key, code in lookup.items():
            if s == key or s.startswith(key):
                return float(code)
        return np.nan
    return f


encode_duration = _ordinal(OPTIONS["sleep_duration"])
encode_quality = _ordinal(OPTIONS["sleep_quality"])
encode_stress = _ordinal(OPTIONS["stress_level"])
encode_bmi = _ordinal(OPTIONS["bmi_category"])       # "Normal (BMI 18.5-24.9)" -> startswith("normal")
encode_steps = _ordinal(OPTIONS["daily_steps"])      # long descriptions start with the band text


def encode_level(x):
    m = re.search(r"(\d)", str(x))
    return float(m.group(1)) if m else np.nan


def group_department(x):
    s = norm_text(x)
    return {"cse": "CSE", "mathematics": "Mathematics", "ece": "ECE", "agriculture": "Agriculture"}.get(s, "Other")


def clean_target(x):
    s = norm_text(x)
    if s.startswith("insomnia"):
        return "Insomnia"
    if "apnea" in s:
        return "Sleep Apnea"
    if s.startswith("no"):
        return "No Sleep Disorder"
    return np.nan


def parse_bp(x):
    """'120/80' -> (120., 80.). Physiologically impossible readings become NaN (imputed later in the pipeline)."""
    m = re.match(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})\s*$", str(x))
    if not m:
        return (np.nan, np.nan)
    s, d = int(m.group(1)), int(m.group(2))
    if s <= d or not (70 <= s <= 220) or not (40 <= d <= 130):
        return (np.nan, np.nan)
    return (float(s), float(d))


def build_features(df):
    """Raw (renamed) survey columns -> model-ready feature frame (one row per respondent)."""
    out = pd.DataFrame(index=df.index)
    out["age"] = pd.to_numeric(df["age"], errors="coerce").clip(18, 30)                 # 2 adults >30 capped
    out["physical_activity"] = pd.to_numeric(df["physical_activity"], errors="coerce").clip(0, 180)
    out["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce").clip(40, 200)
    bp = df["blood_pressure"].map(parse_bp)
    out["systolic"] = bp.map(lambda t: t[0])
    out["diastolic"] = bp.map(lambda t: t[1])
    out["map"] = (out["systolic"] + 2 * out["diastolic"]) / 3                            # mean arterial pressure
    out["pulse_pressure"] = out["systolic"] - out["diastolic"]
    out["sleep_duration"] = df["sleep_duration"].map(encode_duration)
    out["sleep_quality"] = df["sleep_quality"].map(encode_quality)
    out["stress_level"] = df["stress_level"].map(encode_stress)
    out["bmi_category"] = df["bmi_category"].map(encode_bmi)
    out["daily_steps"] = df["daily_steps"].map(encode_steps)
    out["academic_level"] = df["academic_level"].map(encode_level)
    out["gender"] = df["gender"].astype(str).str.strip().str.title()
    out["department"] = df["department"].map(group_department)
    return out[FEATURES]


# ----------------------------------------------------------------------------- decision rules (notebook AND API)
# Ties: with k = 25 neighbours the probabilities are multiples of 4%, so two classes can share the top score.
# A screening aid should not clear someone on a tie, so a disorder wins over "No Sleep Disorder";
# between the two disorders the more common one in the training data (Insomnia, 43% vs 39%) wins.
TIE_ORDER = ["Insomnia", "Sleep Apnea", "No Sleep Disorder"]


def clinical_rule_mask(X):
    """True where Insomnia is clinically inconsistent with the answers: 8+ hours of sleep AND Good/Excellent quality.
    DSM-5 / ICSD-3 insomnia requires dissatisfaction with sleep quantity or quality, which these answers rule out."""
    return ((X["sleep_duration"] == OPTIONS["sleep_duration"].index("8 hours or more"))
            & (X["sleep_quality"] >= OPTIONS["sleep_quality"].index("Good"))).to_numpy()


def apply_clinical_rule(proba, X, classes):
    """Set P(Insomnia) to 0 where clinical_rule_mask is True and rescale the other two classes to sum to 1
    (i.e. the model's probabilities given that it is not insomnia). Rows where the model gave Insomnia 100% are left as they are."""
    proba = np.array(proba, dtype=float)
    ins = list(classes).index("Insomnia")
    rest = 1 - proba[:, ins]
    mask = clinical_rule_mask(X) & (rest > 0)
    proba[mask, ins] = 0
    proba[mask] /= rest[mask, None]
    return proba, mask


def decide(proba, classes):
    """Index of the reported class per row: highest probability, ties broken with TIE_ORDER."""
    proba = np.asarray(proba, dtype=float)
    prio = [TIE_ORDER.index(c) for c in classes]
    out = []
    for row in proba:
        top = np.flatnonzero(np.isclose(row, row.max(), atol=1e-9))
        out.append(min(top, key=lambda k: prio[k]))
    return np.array(out)


def tied_classes(row, classes):
    """Names of the classes that share the highest probability (one name when there is no tie), in TIE_ORDER."""
    top = [c for c, v in zip(classes, row) if np.isclose(v, max(row), atol=1e-9)]
    return sorted(top, key=TIE_ORDER.index)


# ----------------------------------------------------------------------------- analysis helpers
def pretty_params(params):
    """Readable best-parameter dict (drops step prefixes, shows SMOTE by class name)."""
    return {k.split("__")[-1]: (type(v).__name__ if hasattr(v, "fit_resample") else v) for k, v in params.items()}


def _bh_adjust(p):
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out


def association_table(X, y):
    """Feature-vs-target association: Kruskal-Wallis (numeric) / chi-square (categorical, ordinal), with effect sizes."""
    rows = []
    for c in NUM_COLS:
        groups = [X.loc[y == k, c].dropna() for k in sorted(y.unique())]
        H, p = kruskal(*groups); n = sum(len(g) for g in groups)
        rows.append((c, "Kruskal-Wallis", (H - len(groups) + 1) / (n - len(groups)), p))
    for c in ORD_COLS + CAT_COLS:
        ct = pd.crosstab(X[c], y); chi2, p, _, _ = chi2_contingency(ct)
        rows.append((c, "Chi-square", np.sqrt(chi2 / ct.values.sum() / (min(ct.shape) - 1)), p))
    t = pd.DataFrame(rows, columns=["feature", "test", "effect size", "p"])
    t["p (FDR-adjusted)"] = _bh_adjust(t["p"])
    t["significant (5%)"] = t["p (FDR-adjusted)"] < 0.05
    return t.sort_values("p").reset_index(drop=True)


def bootstrap_f1_ci(y_true, y_pred, n=1000, seed=42):
    """95% bootstrap confidence interval of macro-F1."""
    rng = np.random.default_rng(seed); y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    v = [f1_score(y_true[i], y_pred[i], average="macro") for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return tuple(np.percentile(v, [2.5, 97.5]))
