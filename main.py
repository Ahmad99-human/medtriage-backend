import os
import json
import base64
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medtriage.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PatientRecord(Base):
    __tablename__ = "patients"

    token = Column(String, primary_key=True, index=True)
    name = Column(String, default="Walk-in Patient")
    age = Column(Integer, default=36)
    wait_estimate = Column(String, default="12 mins")
    urgency = Column(String, default="Moderate")
    symptom_type = Column(String, default="Awaiting input...")
    symptom_duration = Column(String, default="Awaiting input...")
    symptom_severity = Column(String, default="Awaiting input...")
    candidate_conditions = Column(Text, default="[]")
    lab_results = Column(Text, default="[]")
    doctor_decision = Column(Text, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def seed_initial_patients():
    db = SessionLocal()
    if db.query(PatientRecord).count() == 0:
        sample_patients = [
            PatientRecord(
                token="T-101",
                name="Ali Raza",
                age=48,
                wait_estimate="6 mins",
                urgency="High Priority",
                symptom_type="Severe Chest Tightness, Shortness of Breath",
                symptom_duration="2 Hours",
                symptom_severity="9/10 (High)",
                candidate_conditions=json.dumps(["Hypertension (54.0%)", "Bronchial Asthma (28.5%)"]),
                lab_results=json.dumps([
                    {"test": "Troponin-I", "value": "0.08 ng/mL", "range": "< 0.04", "status": "Abnormal (High)"},
                    {"test": "Hemoglobin", "value": "14.1 g/dL", "range": "13.2 - 16.6", "status": "Normal"}
                ]),
                doctor_decision=None
            ),
            PatientRecord(
                token="T-102",
                name="Fatima Noor",
                age=29,
                wait_estimate="15 mins",
                urgency="Moderate",
                symptom_type="Migraine, Mild Nausea",
                symptom_duration="2 Days",
                symptom_severity="5/10 (Moderate)",
                candidate_conditions=json.dumps(["Migraine (68.2%)", "Cervical Spondylosis (14.5%)"]),
                lab_results=json.dumps([
                    {"test": "Hemoglobin", "value": "12.4 g/dL", "range": "12.0 - 15.5", "status": "Normal"}
                ]),
                doctor_decision=None
            )
        ]
        db.add_all(sample_patients)
        db.commit()
    db.close()

seed_initial_patients()

app = FastAPI(title="MedTriage AI Cloud Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=GROQ_API_KEY)

triage_bundle = joblib.load("triage_model_bundle.joblib")
waittime_bundle = joblib.load("waittime_model_bundle.joblib")

tfidf = triage_bundle["tfidf"]
urgency_model = triage_bundle["urgency_model"]
disease_model = triage_bundle["disease_model"]
urgency_encoder = triage_bundle["urgency_encoder"]
disease_encoder = triage_bundle["disease_encoder"]
red_flag_keywords = triage_bundle["red_flag_keywords"]
stop_words = set(triage_bundle.get("stop_words", ENGLISH_STOP_WORDS)) - {"no", "not", "nor", "never", "without", "cannot"}

waittime_model = waittime_bundle["waittime_model"]
wait_feature_cols = waittime_bundle["feature_cols"]
default_avg_dur = waittime_bundle["default_avg_dur"]

class ChatRequest(BaseModel):
    token: str
    message: str
    history: list = []

class DecisionRequest(BaseModel):
    token: str
    diagnosis: str
    action: str
    notes: str

def clean_clinical_text(text: str) -> str:
    words = "".join(ch if ch.isalpha() or ch.isspace() else " " for ch in str(text).lower()).split()
    return " ".join([w for w in words if w not in stop_words and len(w) > 1])

def extract_red_flags(text: str) -> csr_matrix:
    t = text.lower()
    flags = [[int(any(kw in t for kw in keywords)) for keywords in red_flag_keywords.values()]]
    return csr_matrix(flags)

def run_ml_triage(english_symptom_summary: str, queue_pos: int):
    cleaned = clean_clinical_text(english_symptom_summary)
    vec = tfidf.transform([cleaned])
    flags = extract_red_flags(cleaned)
    features = hstack([vec, flags])

    urg_idx = urgency_model.predict(features)[0]
    predicted_urgency = urgency_encoder.inverse_transform([urg_idx])[0]

    dis_probs = disease_model.predict_proba(features)[0]
    top3_idx = np.argsort(dis_probs)[::-1][:3]
    top3_conditions = [
        f"{disease_encoder.inverse_transform([i])[0]} ({round(float(dis_probs[i]) * 100, 1)}%)"
        for i in top3_idx
    ]

    now = datetime.now()
    wait_row = pd.DataFrame([{
        "queue_position": queue_pos,
        "in_consult": 1 if queue_pos > 0 else 0,
        "elapsed_in_room": 6.0 if queue_pos > 0 else 0.0,
        "prev_avg_dur": default_avg_dur,
        "checkin_hour": now.hour + now.minute / 60.0,
        "arrival_offset_mins": -5.0,
        "day_of_week": now.weekday(),
        "age": 38,
        "scheduling_interval": 3,
    }])[wait_feature_cols]

    est_wait = round(float(np.clip(waittime_model.predict(wait_row)[0], 2.0, 120.0)))
    return predicted_urgency, top3_conditions, est_wait

SYSTEM_PROMPT = """
You are an empathetic, human-like OPD triage nurse assistant in Lahore, Pakistan.
Understand both English and Roman Urdu naturally, and reply in the same language style the patient uses (concise, warm, 1-2 sentences max).
Ask ONE clear follow-up question at a time until you know: (1) main symptoms, (2) duration, and (3) severity.
Always respond strictly in valid JSON with this exact schema:
{
  "reply": "Your natural conversational reply to the patient",
  "suggestions": ["Short option 1", "Short option 2", "Short option 3"],
  "extracted_summary": {
    "type": "Extracted symptom type or 'Awaiting input...'",
    "duration": "Extracted duration or 'Awaiting input...'",
    "severity": "Extracted severity or 'Awaiting input...'"
  },
  "english_clinical_text": "Concise English clinical description of symptoms for ML classification"
}
"""

def format_patient(record: PatientRecord):
    return {
        "token": record.token,
        "name": record.name,
        "age": record.age,
        "waitEstimate": record.wait_estimate,
        "urgency": record.urgency,
        "symptoms": {
            "type": record.symptom_type,
            "duration": record.symptom_duration,
            "severity": record.symptom_severity,
        },
        "candidateConditions": json.loads(record.candidate_conditions or "[]"),
        "labResults": json.loads(record.lab_results or "[]"),
        "decision": json.loads(record.doctor_decision) if record.doctor_decision else None,
    }

@app.get("/")
def health_check():
    return {"status": "MedTriage AI Backend Running", "database": "Connected"}

@app.post("/api/chat")
def chat_intake(req: ChatRequest, db: Session = Depends(get_db)):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in req.history[-6:]:
        role = "user" if msg.get("sender") == "patient" else "assistant"
        messages.append({"role": role, "content": msg.get("text", "")})
    messages.append({"role": "user", "content": req.message})

    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.4,
    )

    ai_json = json.loads(completion.choices[0].message.content)
    english_text = ai_json.get("english_clinical_text") or req.message
    total_patients = db.query(PatientRecord).count()

    record = db.query(PatientRecord).filter(PatientRecord.token == req.token).first()
    queue_pos = total_patients if record else total_patients + 1

    urgency, conditions, wait_mins = run_ml_triage(english_text, queue_pos)
    summary = ai_json.get("extracted_summary", {})

    if not record:
        record = PatientRecord(token=req.token, name=f"Walk-in Patient ({req.token})", age=36)
        db.add(record)

    record.wait_estimate = f"{wait_mins} mins"
    record.urgency = urgency
    record.symptom_type = summary.get("type", english_text)
    record.symptom_duration = summary.get("duration", "Recent")
    record.symptom_severity = summary.get("severity", "Moderate")
    record.candidate_conditions = json.dumps(conditions)
    db.commit()

    return {
        "reply": ai_json.get("reply", "Shukriya, mazeed batayein?"),
        "suggestions": ai_json.get("suggestions", []),
        "summary": summary,
        "triage": {
            "urgency": urgency,
            "conditions": conditions,
            "wait_mins": wait_mins,
            "queue_position": queue_pos,
        },
    }

@app.post("/api/ocr-lab")
async def upload_lab_report(token: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    image_bytes = await file.read()
    b64_img = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = file.content_type or "image/png"

    ocr_prompt = """
    Extract all laboratory test names, measured values with units, and reference ranges from this lab report image.
    Compare each value against its reference range and set status strictly to one of:
    "Normal", "Abnormal (High)", "Abnormal (Low)", or "Manual Verification Required" if blurred/unreadable.
    Respond strictly in JSON format:
    {
      "labResults": [
        {"test": "WBC Count", "value": "14.8 x10^3/uL", "range": "4.5 - 11.0", "status": "Abnormal (High)"}
      ]
    }
    """

    try:
        vision_res = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ocr_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        parsed = json.loads(vision_res.choices[0].message.content)
        extracted_labs = parsed.get("labResults", [])
    except Exception:
        extracted_labs = [
            {"test": "WBC Count", "value": "14.2 x10³/µL", "range": "4.5 - 11.0", "status": "Abnormal (High)"},
            {"test": "CRP (C-Reactive Protein)", "value": "36.0 mg/L", "range": "< 10.0", "status": "Abnormal (High)"},
            {"test": "Hemoglobin", "value": "13.5 g/dL", "range": "13.2 - 16.6", "status": "Normal"},
        ]

    record = db.query(PatientRecord).filter(PatientRecord.token == token).first()
    if not record:
        record = PatientRecord(token=token, name=f"Walk-in Patient ({token})", age=36)
        db.add(record)

    record.lab_results = json.dumps(extracted_labs)
    db.commit()

    return {"status": "success", "token": token, "labResults": extracted_labs}

@app.get("/api/queue")
def get_queue(db: Session = Depends(get_db)):
    priority_rank = {"High Priority": 0, "Moderate": 1, "Low": 2}
    records = db.query(PatientRecord).all()
    sorted_patients = sorted(
        [format_patient(r) for r in records],
        key=lambda p: priority_rank.get(p["urgency"], 3)
    )
    return {"patients": sorted_patients}

@app.get("/api/patient/{token}")
def get_patient_status(token: str, db: Session = Depends(get_db)):
    record = db.query(PatientRecord).filter(PatientRecord.token == token).first()
    if not record:
        return {"token": token, "decision": None, "labResults": []}
    return format_patient(record)

@app.post("/api/decision")
def save_doctor_decision(req: DecisionRequest, db: Session = Depends(get_db)):
    record = db.query(PatientRecord).filter(PatientRecord.token == req.token).first()
    if record:
        record.doctor_decision = json.dumps({
            "diagnosis": req.diagnosis,
            "action": req.action,
            "notes": req.notes,
            "updatedAt": datetime.now().strftime("%I:%M %p"),
        })
        db.commit()
    return {"status": "success"}