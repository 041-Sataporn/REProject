import glob
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# ตั้งค่า
# ----------------------------------------------------------------------------
MODEL_FILE = "fraud_model.joblib"
SCALER_FILE = "scaler.joblib"  # ใช้เมื่อ scaler ไม่ได้อยู่ในไฟล์โมเดล

DEFAULT_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ใช้เฉพาะเมื่อไฟล์โมเดลไม่ได้เก็บ "metrics" ไว้ (ใส่ค่าจริงจากชุดทดสอบ)
MANUAL_METRICS = {}  # เช่น {"accuracy": 0.9992, "precision": 1.0, "recall": 0.8, "f1": 0.8889, "auc": 0.9493}

DEVELOPERS = [
    ("039", "ศักดิโชติ แตงโสภา"),
    ("041", "สถาพร ขวาธิจักร"),
    ("044", "อติชาต พันธุ์ขะวงษ์"),
]

st.set_page_config(page_title="ระบบตรวจจับธุรกรรมฉ้อโกง", page_icon="🛡️", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Sarabun', sans-serif; }
    .block-container { max-width: 820px; padding-top: 2.2rem; }
    .title { text-align: center; margin-bottom: .2rem; font-weight: 700; font-size: 2rem; color: #0F172A; }
    .subtitle { text-align: center; color: #64748B; margin-bottom: 1.6rem; }
    .result { border-radius: 12px; padding: 1.1rem 1.3rem; margin-top: .8rem; border: 1px solid; }
    .result.ok   { background: #F0FDF9; border-color: #99E6D3; color: #0F5132; }
    .result.bad  { background: #FEF2F2; border-color: #FECACA; color: #7F1D1D; }
    .result h3 { margin: 0 0 .2rem 0; font-size: 1.3rem; }
    .footer { text-align: center; color: #64748B; font-size: .92rem; line-height: 1.7; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #0F766E; }
    hr { margin: 1.8rem 0 1.2rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# โหลดโมเดล
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_artifacts():
    path = MODEL_FILE
    if not os.path.exists(path):
        cands = [p for p in glob.glob("*.joblib") if os.path.basename(p) != SCALER_FILE]
        if not cands:
            raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {MODEL_FILE}")
        path = cands[0]
    obj = joblib.load(path)

    scaler = obj.get("scaler") if isinstance(obj, dict) else None
    if scaler is None and os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
    return obj, scaler, path


def get_features(obj):
    if isinstance(obj, dict) and obj.get("features"):
        return list(obj["features"])
    return DEFAULT_FEATURES


def get_metrics(obj):
    if isinstance(obj, dict) and isinstance(obj.get("metrics"), dict):
        return obj["metrics"]
    return MANUAL_METRICS


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def predict_proba(obj, scaler, df):
    """คืนค่าความน่าจะเป็นที่เป็นธุรกรรมฉ้อโกง (Fraud) ของทุกแถว"""
    features = get_features(obj)
    X = df[features].astype(float)

    # แบบที่ 1: โมเดล Neural Network ที่เก็บเป็นน้ำหนักแต่ละชั้น (bundle จากโน้ตบุ๊ก)
    if isinstance(obj, dict) and "layers" in obj:
        a = scaler.transform(X) if scaler is not None else X.values
        a = np.asarray(a, dtype=float)
        for layer in obj["layers"]:
            a = a @ layer["W"] + layer["b"]
            act = layer.get("activation", "linear")
            if act == "relu":
                a = np.maximum(a, 0)
            elif act == "sigmoid":
                a = _sigmoid(a)
            elif act == "tanh":
                a = np.tanh(a)
        return a.ravel()

    # แบบที่ 2: โมเดล/Pipeline ของ scikit-learn
    model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    Xm = scaler.transform(X) if scaler is not None else X
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xm)[:, 1]
    if hasattr(model, "decision_function"):
        return _sigmoid(model.decision_function(Xm))
    return model.predict(Xm).astype(float)


def get_threshold(obj):
    return float(obj.get("threshold", 0.5)) if isinstance(obj, dict) else 0.5


def show_result(prob, threshold):
    is_fraud = prob >= threshold
    cls, icon, text = (
        ("bad", "⚠️", "น่าสงสัยว่าเป็นธุรกรรมฉ้อโกง") if is_fraud else ("ok", "✅", "ธุรกรรมปกติ")
    )
    st.markdown(
        f'<div class="result {cls}"><h3>{icon} {text}</h3>'
        f"โอกาสเป็นธุรกรรมฉ้อโกง: <b>{prob * 100:.2f}%</b> "
        f"(เกณฑ์ตัดสิน {threshold * 100:.0f}%)</div>",
        unsafe_allow_html=True,
    )
    st.progress(float(min(max(prob, 0.0), 1.0)))


# ----------------------------------------------------------------------------
# ส่วนหัว
# ----------------------------------------------------------------------------
st.markdown('<div class="title">🛡️ ระบบตรวจจับธุรกรรมบัตรเครดิตที่ฉ้อโกง</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Credit Card Fraud Detection · ทำนายด้วย Neural Network</div>',
    unsafe_allow_html=True,
)

try:
    obj, scaler, model_path = load_artifacts()
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.info(f"ตรวจสอบว่าไฟล์ `{MODEL_FILE}` (และ `{SCALER_FILE}` ถ้ามี) อยู่โฟลเดอร์เดียวกับ app.py")
    st.stop()

features = get_features(obj)
threshold = get_threshold(obj)

if isinstance(obj, dict) and "layers" in obj and scaler is None:
    st.warning(
        f"ไม่พบ scaler — โมเดล Neural Network ต้องใช้ `{SCALER_FILE}` ร่วมด้วย "
        "ไม่เช่นนั้นผลทำนายจะไม่ถูกต้อง"
    )

# ----------------------------------------------------------------------------
# ส่วนใช้งาน
# ----------------------------------------------------------------------------
tab_manual, tab_csv = st.tabs(["กรอกข้อมูลทีละรายการ", "อัปโหลดไฟล์ CSV"])

with tab_manual:
    st.caption("กรอกค่าของธุรกรรม แล้วกด “ตรวจสอบ” (ค่า V1–V28 เป็นฟีเจอร์ที่ผ่านการแปลงแล้วจากชุดข้อมูล)")
    c1, c2 = st.columns(2)
    values = {}
    if "Time" in features:
        values["Time"] = c1.number_input("Time (วินาทีนับจากธุรกรรมแรก)", value=0.0, step=1.0)
    if "Amount" in features:
        values["Amount"] = c2.number_input("Amount (จำนวนเงิน)", value=0.0, min_value=0.0, step=1.0)

    v_feats = [f for f in features if f not in ("Time", "Amount")]
    with st.expander(f"ฟีเจอร์ {', '.join([v_feats[0], '…', v_feats[-1]]) if v_feats else ''} ({len(v_feats)} ค่า)"):
        cols = st.columns(4)
        for i, f in enumerate(v_feats):
            values[f] = cols[i % 4].number_input(f, value=0.0, format="%.4f", key=f"in_{f}")

    if st.button("ตรวจสอบธุรกรรม", type="primary", use_container_width=True):
        row = pd.DataFrame([values])[features]
        prob = float(predict_proba(obj, scaler, row)[0])
        show_result(prob, threshold)

with tab_csv:
    st.caption(f"ไฟล์ต้องมีคอลัมน์: {', '.join(features[:3])} … {features[-1]} (ไม่ต้องมีคอลัมน์ Class)")
    up = st.file_uploader("เลือกไฟล์ CSV", type=["csv"])
    if up is not None:
        data = pd.read_csv(up)
        missing = [f for f in features if f not in data.columns]
        if missing:
            st.error(f"ไฟล์ขาดคอลัมน์: {', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}")
        else:
            work = data.copy()
            work[features] = work[features].fillna(work[features].median())
            probs = predict_proba(obj, scaler, work)
            out = data.copy()
            out["โอกาสฉ้อโกง (%)"] = (probs * 100).round(2)
            out["ผลการทำนาย"] = np.where(probs >= threshold, "น่าสงสัย", "ปกติ")

            n_bad = int((probs >= threshold).sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("จำนวนธุรกรรม", f"{len(out):,}")
            m2.metric("น่าสงสัย", f"{n_bad:,}")
            m3.metric("สัดส่วนที่น่าสงสัย", f"{n_bad / len(out) * 100:.2f}%")

            st.dataframe(out, use_container_width=True, height=300)
            st.download_button(
                "ดาวน์โหลดผลลัพธ์ (CSV)",
                out.to_csv(index=False).encode("utf-8-sig"),
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

# ----------------------------------------------------------------------------
# ค่าความแม่นยำของระบบ
# ----------------------------------------------------------------------------
st.markdown("---")
st.subheader("ความแม่นยำของระบบ")

metrics = get_metrics(obj)
if metrics:
    labels = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("auc", "ROC-AUC"),
    ]
    shown = [(k, n) for k, n in labels if k in metrics]
    cols = st.columns(len(shown))
    for col, (k, name) in zip(cols, shown):
        col.metric(name, f"{float(metrics[k]) * 100:.2f}%")

    st.caption(
        "ประเมินจากชุดทดสอบ (20% ของข้อมูล, 2,392 รายการ, มี Fraud 10 รายการ) · "
        "Accuracy สูงได้ง่ายเพราะข้อมูลส่วนใหญ่เป็นธุรกรรมปกติ (การทายว่า “ปกติ” ทั้งหมดก็ได้ ≈ 99.58%) "
        "จึงควรดู Precision / Recall ประกอบ"
    )
    if float(metrics.get("recall", 1)) == 0:
        st.warning("โมเดลชุดนี้ยังตรวจจับธุรกรรมฉ้อโกงในชุดทดสอบไม่ได้ (Recall = 0%) ควรใช้เพื่อการศึกษาเท่านั้น")
else:
    st.info("ไม่พบค่าความแม่นยำในไฟล์โมเดล — ใส่ค่าใน MANUAL_METRICS ที่ต้นไฟล์ app.py")

# ----------------------------------------------------------------------------
# ผู้พัฒนา
# ----------------------------------------------------------------------------
st.markdown("---")
dev_html = "<br>".join(f"<b>{sid}</b> {name}" for sid, name in DEVELOPERS)
st.markdown(f'<div class="footer">ผู้พัฒนา<br>{dev_html}</div>', unsafe_allow_html=True)
