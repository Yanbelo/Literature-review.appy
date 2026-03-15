
import os
import sqlite3
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="LR-HINDRI Deployment App",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_NAME = os.getenv("LR_HINDRI_DB", "lr_hindri.db")
ADMIN_USERNAME = os.getenv("LR_HINDRI_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("LR_HINDRI_ADMIN_PASSWORD", "admin123")
APP_TITLE = os.getenv(
    "LR_HINDRI_APP_TITLE",
    "LR-HINDRI Advanced Streamlit Platform"
)

# =========================================================
# DATABASE
# =========================================================
@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


conn = get_connection()


def init_db():
    query = """
    CREATE TABLE IF NOT EXISTS participants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_id INTEGER,
        visit_type TEXT,
        study_id TEXT,
        visit_date TEXT,
        assessor_id TEXT,
        community TEXT,
        clinic_area TEXT,
        contact TEXT,
        consent TEXT,
        sex TEXT,
        pregnancy TEXT,
        age INTEGER,
        education TEXT,
        employment TEXT,
        clinic_distance TEXT,
        transport TEXT,
        clinic_visits TEXT,
        previous_high_bp INTEGER,
        bp_checked_12m INTEGER,
        family_history_htn INTEGER,
        on_antihypertensives INTEGER,
        missed_bp_meds INTEGER,
        extra_salt INTEGER,
        processed_food_freq TEXT,
        physical_activity TEXT,
        tobacco_use INTEGER,
        alcohol_use TEXT,
        ever_overweight INTEGER,
        prolonged_sitting INTEGER,
        known_diabetes INTEGER,
        high_blood_sugar INTEGER,
        known_kidney_disease INTEGER,
        known_cvd INTEGER,
        family_history_cvd INTEGER,
        weight_kg REAL,
        height_cm REAL,
        waist_cm REAL,
        bmi REAL,
        central_obesity INTEGER,
        headaches INTEGER,
        dizziness INTEGER,
        blurred_vision INTEGER,
        chest_pain INTEGER,
        shortness_breath INTEGER,
        palpitations INTEGER,
        swollen_feet INTEGER,
        no_symptoms INTEGER,
        access_clinic TEXT,
        missed_care_transport INTEGER,
        medicine_runout INTEGER,
        food_cost_barrier INTEGER,
        frequent_stress INTEGER,
        traditional_medicine INTEGER,
        bp_machine_access TEXT,
        sbp1 REAL,
        dbp1 REAL,
        sbp2 REAL,
        dbp2 REAL,
        sbp3 REAL,
        dbp3 REAL,
        mean_sbp REAL,
        mean_dbp REAL,
        bp_status TEXT,
        risk_total_score INTEGER,
        risk_category TEXT,
        referral_decision TEXT,
        measured_hypertension INTEGER,
        created_at TEXT,
        updated_at TEXT
    )
    """
    conn.execute(query)
    conn.commit()


def save_record(record: dict):
    pd.DataFrame([record]).to_sql("participants", conn, if_exists="append", index=False)


def load_data() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM participants ORDER BY id DESC", conn)


def load_record_by_id(record_id: int) -> pd.DataFrame:
    query = "SELECT * FROM participants WHERE id = ?"
    return pd.read_sql(query, conn, params=(record_id,))


def update_record(record_id: int, record: dict):
    cols = [k for k in record.keys()]
    set_clause = ", ".join([f"{col}=?" for col in cols])
    values = [record[col] for col in cols]
    values.append(record_id)
    query = f"UPDATE participants SET {set_clause} WHERE id=?"
    conn.execute(query, values)
    conn.commit()


def delete_record(record_id: int):
    conn.execute("DELETE FROM participants WHERE id = ?", (record_id,))
    conn.commit()


def delete_all_data():
    conn.execute("DELETE FROM participants")
    conn.commit()


init_db()

# =========================================================
# HELPERS
# =========================================================
def bool_to_int(value: bool) -> int:
    return 1 if value else 0


def int_to_bool(value) -> bool:
    try:
        return bool(int(value))
    except Exception:
        return False


def compute_bmi(weight_kg: float, height_m: float) -> float:
    if height_m <= 0:
        return 0.0
    return round(weight_kg / (height_m ** 2), 2)


def central_obesity(sex: str, waist_cm: float) -> bool:
    if sex == "Male":
        return waist_cm >= 94
    if sex == "Female":
        return waist_cm >= 80
    return False


def average_bp(sbp2: float, sbp3: float, dbp2: float, dbp3: float):
    mean_sbp = round((sbp2 + sbp3) / 2, 1)
    mean_dbp = round((dbp2 + dbp3) / 2, 1)
    return mean_sbp, mean_dbp


def bp_classification(mean_sbp: float, mean_dbp: float) -> str:
    if mean_sbp >= 180 or mean_dbp >= 120:
        return "Hypertensive crisis"
    if mean_sbp >= 140 or mean_dbp >= 90:
        return "Hypertension"
    if mean_sbp >= 130 or mean_dbp >= 85:
        return "Elevated / high-normal"
    return "Normal"


def risk_category(score: int) -> str:
    if score >= 22:
        return "Very high risk"
    if score >= 15:
        return "High risk"
    if score >= 8:
        return "Moderate risk"
    return "Low risk"


def referral_decision(score: int, mean_sbp: float, mean_dbp: float, severe_symptoms: bool) -> str:
    if mean_sbp >= 180 or mean_dbp >= 120 or severe_symptoms:
        return "Urgent same-day referral"
    if score >= 22 or mean_sbp >= 140 or mean_dbp >= 90:
        return "Refer to primary care / clinician"
    if score >= 8 or mean_sbp >= 130 or mean_dbp >= 85:
        return "Counsel, monitor, and follow up in 1–3 months"
    return "Preventive counselling and annual screening"


def compute_score(data: dict) -> int:
    score = 0

    age = data["age"]
    if 35 <= age <= 44:
        score += 1
    elif 45 <= age <= 54:
        score += 2
    elif 55 <= age <= 64:
        score += 3
    elif age >= 65:
        score += 4

    if data["sex"] == "Male":
        score += 1
    if data["family_history_htn"]:
        score += 2
    if data["tobacco_use"]:
        score += 2
    if data["harmful_alcohol"]:
        score += 1
    if data["physical_inactivity"]:
        score += 2
    if data["salty_foods"]:
        score += 2

    bmi = data["bmi"]
    if 25 <= bmi < 30:
        score += 1
    elif bmi >= 30:
        score += 3

    if data["central_obesity"]:
        score += 2
    if data["known_diabetes"]:
        score += 3
    if data["known_kidney_disease"]:
        score += 3
    if data["known_cvd"]:
        score += 3
    if data["previous_high_bp"]:
        score += 3
    if data["on_antihypertensives"]:
        score += 1
    if data["missed_bp_meds"]:
        score += 2

    if data["headaches"]:
        score += 1
    if data["dizziness"]:
        score += 1
    if data["blurred_vision"]:
        score += 1
    if data["chest_pain"]:
        score += 2
    if data["sob_palpitations"]:
        score += 2

    if data["missed_care_transport"]:
        score += 1
    if data["medicine_runout"]:
        score += 2
    if data["no_bp_check_12m"]:
        score += 2
    if data["poor_access_clinic"]:
        score += 1
    if data["frequent_stress"]:
        score += 1

    return score


def df_to_excel_bytes(sheet_dict: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheet_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output.getvalue()


def make_record_dict(
    visit_type,
    parent_id,
    study_id,
    visit_date,
    assessor_id,
    community,
    clinic_area,
    contact,
    consent,
    sex,
    pregnancy,
    age,
    education,
    employment,
    clinic_distance,
    transport,
    clinic_visits,
    previous_high_bp,
    bp_checked_12m,
    family_history_htn,
    on_antihypertensives,
    missed_bp_meds,
    extra_salt,
    processed_food_freq,
    physical_activity,
    tobacco_use,
    alcohol_use,
    ever_overweight,
    prolonged_sitting,
    known_diabetes,
    high_blood_sugar,
    known_kidney_disease,
    known_cvd,
    family_history_cvd,
    weight_kg,
    height_cm,
    waist_cm,
    headaches,
    dizziness,
    blurred_vision,
    chest_pain,
    shortness_breath,
    palpitations,
    swollen_feet,
    no_symptoms,
    access_clinic,
    missed_care_transport,
    medicine_runout,
    food_cost_barrier,
    frequent_stress,
    traditional_medicine,
    bp_machine_access,
    sbp1,
    dbp1,
    sbp2,
    dbp2,
    sbp3,
    dbp3,
):
    height_m = height_cm / 100.0
    bmi = compute_bmi(weight_kg, height_m)
    is_central_obesity = central_obesity(sex, waist_cm)
    mean_sbp, mean_dbp = average_bp(sbp2, sbp3, dbp2, dbp3)
    bp_status = bp_classification(mean_sbp, mean_dbp)

    physical_inactivity = physical_activity in ["Never", "1–2 times/week"]
    salty_foods = processed_food_freq in ["3–4 times/week", "Almost daily"]
    harmful_alcohol = alcohol_use == "Frequent"
    poor_access_clinic = access_clinic == "No"
    no_bp_check_12m = not bp_checked_12m
    sob_palpitations = shortness_breath or palpitations
    measured_hypertension = int(mean_sbp >= 140 or mean_dbp >= 90)

    if no_symptoms:
        headaches = False
        dizziness = False
        blurred_vision = False
        chest_pain = False
        shortness_breath = False
        palpitations = False
        swollen_feet = False
        sob_palpitations = False

    score_data = {
        "age": age,
        "sex": sex,
        "family_history_htn": family_history_htn,
        "tobacco_use": tobacco_use,
        "harmful_alcohol": harmful_alcohol,
        "physical_inactivity": physical_inactivity,
        "salty_foods": salty_foods,
        "bmi": bmi,
        "central_obesity": is_central_obesity,
        "known_diabetes": known_diabetes,
        "known_kidney_disease": known_kidney_disease,
        "known_cvd": known_cvd,
        "previous_high_bp": previous_high_bp,
        "on_antihypertensives": on_antihypertensives,
        "missed_bp_meds": missed_bp_meds,
        "headaches": headaches,
        "dizziness": dizziness,
        "blurred_vision": blurred_vision,
        "chest_pain": chest_pain,
        "sob_palpitations": sob_palpitations,
        "missed_care_transport": missed_care_transport,
        "medicine_runout": medicine_runout,
        "no_bp_check_12m": no_bp_check_12m,
        "poor_access_clinic": poor_access_clinic,
        "frequent_stress": frequent_stress,
    }

    total_score = compute_score(score_data)
    category = risk_category(total_score)
    severe_symptoms = chest_pain and (mean_sbp >= 140 or mean_dbp >= 90)
    action = referral_decision(total_score, mean_sbp, mean_dbp, severe_symptoms)
    now_ts = datetime.now().isoformat()

    return {
        "parent_id": parent_id,
        "visit_type": visit_type,
        "study_id": study_id,
        "visit_date": str(visit_date),
        "assessor_id": assessor_id,
        "community": community,
        "clinic_area": clinic_area,
        "contact": contact,
        "consent": consent,
        "sex": sex,
        "pregnancy": pregnancy,
        "age": age,
        "education": education,
        "employment": employment,
        "clinic_distance": clinic_distance,
        "transport": transport,
        "clinic_visits": clinic_visits,
        "previous_high_bp": 1 if previous_high_bp else 0,
        "bp_checked_12m": 1 if bp_checked_12m else 0,
        "family_history_htn": 1 if family_history_htn else 0,
        "on_antihypertensives": 1 if on_antihypertensives else 0,
        "missed_bp_meds": 1 if missed_bp_meds else 0,
        "extra_salt": 1 if extra_salt else 0,
        "processed_food_freq": processed_food_freq,
        "physical_activity": physical_activity,
        "tobacco_use": 1 if tobacco_use else 0,
        "alcohol_use": alcohol_use,
        "ever_overweight": 1 if ever_overweight else 0,
        "prolonged_sitting": 1 if prolonged_sitting else 0,
        "known_diabetes": 1 if known_diabetes else 0,
        "high_blood_sugar": 1 if high_blood_sugar else 0,
        "known_kidney_disease": 1 if known_kidney_disease else 0,
        "known_cvd": 1 if known_cvd else 0,
        "family_history_cvd": 1 if family_history_cvd else 0,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "waist_cm": waist_cm,
        "bmi": bmi,
        "central_obesity": 1 if is_central_obesity else 0,
        "headaches": 1 if headaches else 0,
        "dizziness": 1 if dizziness else 0,
        "blurred_vision": 1 if blurred_vision else 0,
        "chest_pain": 1 if chest_pain else 0,
        "shortness_breath": 1 if shortness_breath else 0,
        "palpitations": 1 if palpitations else 0,
        "swollen_feet": 1 if swollen_feet else 0,
        "no_symptoms": 1 if no_symptoms else 0,
        "access_clinic": access_clinic,
        "missed_care_transport": 1 if missed_care_transport else 0,
        "medicine_runout": 1 if medicine_runout else 0,
        "food_cost_barrier": 1 if food_cost_barrier else 0,
        "frequent_stress": 1 if frequent_stress else 0,
        "traditional_medicine": 1 if traditional_medicine else 0,
        "bp_machine_access": bp_machine_access,
        "sbp1": sbp1,
        "dbp1": dbp1,
        "sbp2": sbp2,
        "dbp2": dbp2,
        "sbp3": sbp3,
        "dbp3": dbp3,
        "mean_sbp": mean_sbp,
        "mean_dbp": mean_dbp,
        "bp_status": bp_status,
        "risk_total_score": total_score,
        "risk_category": category,
        "referral_decision": action,
        "measured_hypertension": measured_hypertension,
        "created_at": now_ts,
        "updated_at": now_ts,
    }


def render_metrics_card(df):
    total_participants = len(df)
    measured_htn_pct = round(df["measured_hypertension"].mean() * 100, 2) if len(df) else 0
    high_risk_pct = round(df["risk_category"].isin(["High risk", "Very high risk"]).mean() * 100, 2) if len(df) else 0
    obesity_pct = round((df["bmi"] >= 30).mean() * 100, 2) if len(df) else 0

    a, b, c, d = st.columns(4)
    a.metric("Participants", total_participants)
    b.metric("Measured Hypertension (%)", measured_htn_pct)
    c.metric("High / Very High Risk (%)", high_risk_pct)
    d.metric("Obesity (%)", obesity_pct)


if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

st.sidebar.title("LR-HINDRI")
st.sidebar.write("Deployment-ready hypertension risk platform")

all_data_sidebar = load_data()
community_options = ["All"]
if not all_data_sidebar.empty:
    community_options += sorted([str(x) for x in all_data_sidebar["community"].dropna().unique().tolist()])

risk_options = ["All", "Low risk", "Moderate risk", "High risk", "Very high risk"]
sex_options = ["All", "Male", "Female"]

selected_community_sidebar = st.sidebar.selectbox("Filter community", community_options)
selected_risk_sidebar = st.sidebar.selectbox("Filter risk category", risk_options)
selected_sex_sidebar = st.sidebar.selectbox("Filter sex", sex_options)
age_min_sidebar, age_max_sidebar = st.sidebar.slider("Age range", 18, 120, (18, 120))

st.title(APP_TITLE)
st.caption("Limpopo Rural Hypertension and Integrated NCD Risk Index")

tabs = st.tabs([
    "Participant Entry",
    "Dashboard",
    "Community Summary",
    "Records Manager",
    "Data Export",
    "ML Calibration",
    "Admin",
])


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    if filtered.empty:
        return filtered
    if selected_community_sidebar != "All":
        filtered = filtered[filtered["community"].astype(str) == selected_community_sidebar]
    if selected_risk_sidebar != "All":
        filtered = filtered[filtered["risk_category"] == selected_risk_sidebar]
    if selected_sex_sidebar != "All":
        filtered = filtered[filtered["sex"] == selected_sex_sidebar]
    filtered = filtered[(filtered["age"] >= age_min_sidebar) & (filtered["age"] <= age_max_sidebar)]
    return filtered


with tabs[0]:
    st.subheader("Participant Entry")

    with st.form("participant_form"):
        v1, v2, v3 = st.columns(3)
        with v1:
            visit_type = st.selectbox("Visit type", ["baseline", "follow-up"])
        with v2:
            parent_id = st.number_input("Parent ID for follow-up (0 if none)", min_value=0, value=0)
        with v3:
            visit_date = st.date_input("Visit date", value=datetime.today())

        c1, c2, c3 = st.columns(3)
        with c1:
            study_id = st.text_input("Study ID")
            assessor_id = st.text_input("Assessor ID")
            community = st.text_input("Community / Village")
        with c2:
            clinic_area = st.text_input("Clinic catchment area")
            consent = st.selectbox("Consent obtained", ["Yes", "No"])
            contact = st.text_input("Contact number")
        with c3:
            sex = st.selectbox("Sex", ["Male", "Female"])
            pregnancy = st.selectbox("Pregnancy status", ["No", "Yes", "Not applicable"])
            age = st.number_input("Age", min_value=18, max_value=120, value=35)

        d1, d2, d3 = st.columns(3)
        with d1:
            education = st.selectbox("Education", ["No formal schooling", "Primary", "Secondary", "Tertiary"])
            employment = st.selectbox("Employment", ["Unemployed", "Informal work", "Formal employment", "Pension/social grant only"])
        with d2:
            clinic_distance = st.selectbox("Distance to nearest clinic", ["<5 km", "5–10 km", ">10 km"])
            transport = st.selectbox("Transport available", ["Yes", "No"])
        with d3:
            clinic_visits = st.selectbox("Clinic attendance in past year", ["None", "1–2 times", "3–5 times", ">5 times"])

        h1, h2, h3 = st.columns(3)
        with h1:
            previous_high_bp = st.checkbox("Ever told BP is high")
            bp_checked_12m = st.checkbox("BP checked in past 12 months")
            family_history_htn = st.checkbox("Family history of hypertension")
            on_antihypertensives = st.checkbox("Currently on BP medication")
        with h2:
            missed_bp_meds = st.checkbox("Missed BP medication in past 2 weeks")
            extra_salt = st.checkbox("Adds extra salt to food")
            processed_food_freq = st.selectbox("Processed / salty foods", ["Rarely", "1–2 times/week", "3–4 times/week", "Almost daily"])
            physical_activity = st.selectbox("Physical activity ≥30 min", ["Never", "1–2 times/week", "3–4 times/week", "5+ times/week"])
        with h3:
            tobacco_use = st.checkbox("Current tobacco use")
            alcohol_use = st.selectbox("Alcohol use", ["None", "Occasional", "Frequent"])
            ever_overweight = st.checkbox("Ever told overweight/obese")
            prolonged_sitting = st.checkbox("Mostly sedentary / prolonged sitting")

        n1, n2, n3 = st.columns(3)
        with n1:
            known_diabetes = st.checkbox("Known diabetes")
            high_blood_sugar = st.checkbox("Ever told high blood sugar")
            known_kidney_disease = st.checkbox("Known kidney disease")
        with n2:
            known_cvd = st.checkbox("Known heart disease / stroke / clinician-diagnosed chest pain")
            family_history_cvd = st.checkbox("Family history of diabetes / CVD / stroke")
        with n3:
            weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.1)
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=165.0, step=0.1)
            waist_cm = st.number_input("Waist circumference (cm)", min_value=40.0, max_value=200.0, value=85.0, step=0.1)

        s1, s2, s3 = st.columns(3)
        with s1:
            headaches = st.checkbox("Frequent headaches")
            dizziness = st.checkbox("Dizziness")
            blurred_vision = st.checkbox("Blurred vision")
        with s2:
            chest_pain = st.checkbox("Chest pain")
            shortness_breath = st.checkbox("Shortness of breath")
            palpitations = st.checkbox("Palpitations")
        with s3:
            swollen_feet = st.checkbox("Swelling of feet")
            no_symptoms = st.checkbox("None of the above")

        r1, r2, r3 = st.columns(3)
        with r1:
            access_clinic = st.selectbox("Regular access to clinic/pharmacy", ["Yes", "No"])
            missed_care_transport = st.checkbox("Missed clinic visits due to transport cost")
        with r2:
            medicine_runout = st.checkbox("Ran out of hypertension medicine before refill")
            food_cost_barrier = st.checkbox("Healthy diet limited by food cost")
        with r3:
            frequent_stress = st.checkbox("Frequent psychosocial stress")
            traditional_medicine = st.checkbox("Uses traditional medicine for BP/diabetes")
            bp_machine_access = st.selectbox("BP machine available at home/community", ["Yes", "No"])

        bp1, bp2, bp3 = st.columns(3)
        with bp1:
            sbp1 = st.number_input("SBP 1", min_value=60.0, max_value=260.0, value=130.0)
            dbp1 = st.number_input("DBP 1", min_value=30.0, max_value=180.0, value=80.0)
        with bp2:
            sbp2 = st.number_input("SBP 2", min_value=60.0, max_value=260.0, value=128.0)
            dbp2 = st.number_input("DBP 2", min_value=30.0, max_value=180.0, value=82.0)
        with bp3:
            sbp3 = st.number_input("SBP 3", min_value=60.0, max_value=260.0, value=129.0)
            dbp3 = st.number_input("DBP 3", min_value=30.0, max_value=180.0, value=81.0)

        submitted = st.form_submit_button("Save assessment")

    if submitted:
        record = make_record_dict(
            visit_type=visit_type,
            parent_id=(parent_id if parent_id != 0 else None),
            study_id=study_id,
            visit_date=visit_date,
            assessor_id=assessor_id,
            community=community,
            clinic_area=clinic_area,
            contact=contact,
            consent=consent,
            sex=sex,
            pregnancy=pregnancy,
            age=age,
            education=education,
            employment=employment,
            clinic_distance=clinic_distance,
            transport=transport,
            clinic_visits=clinic_visits,
            previous_high_bp=previous_high_bp,
            bp_checked_12m=bp_checked_12m,
            family_history_htn=family_history_htn,
            on_antihypertensives=on_antihypertensives,
            missed_bp_meds=missed_bp_meds,
            extra_salt=extra_salt,
            processed_food_freq=processed_food_freq,
            physical_activity=physical_activity,
            tobacco_use=tobacco_use,
            alcohol_use=alcohol_use,
            ever_overweight=ever_overweight,
            prolonged_sitting=prolonged_sitting,
            known_diabetes=known_diabetes,
            high_blood_sugar=high_blood_sugar,
            known_kidney_disease=known_kidney_disease,
            known_cvd=known_cvd,
            family_history_cvd=family_history_cvd,
            weight_kg=weight_kg,
            height_cm=height_cm,
            waist_cm=waist_cm,
            headaches=headaches,
            dizziness=dizziness,
            blurred_vision=blurred_vision,
            chest_pain=chest_pain,
            shortness_breath=shortness_breath,
            palpitations=palpitations,
            swollen_feet=swollen_feet,
            no_symptoms=no_symptoms,
            access_clinic=access_clinic,
            missed_care_transport=missed_care_transport,
            medicine_runout=medicine_runout,
            food_cost_barrier=food_cost_barrier,
            frequent_stress=frequent_stress,
            traditional_medicine=traditional_medicine,
            bp_machine_access=bp_machine_access,
            sbp1=sbp1,
            dbp1=dbp1,
            sbp2=sbp2,
            dbp2=dbp2,
            sbp3=sbp3,
            dbp3=dbp3,
        )
        save_record(record)
        st.success("Assessment saved successfully.")
        st.metric("Risk Score", record["risk_total_score"])
        st.metric("Risk Category", record["risk_category"])
        st.write(f"**BP classification:** {record['bp_status']}")
        st.write(f"**Referral decision:** {record['referral_decision']}")

with tabs[1]:
    st.subheader("Dashboard")
    df = apply_filters(load_data())
    if df.empty:
        st.info("No data available for the selected filters.")
    else:
        render_metrics_card(df)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            (
                df["risk_category"]
                .value_counts()
                .reindex(["Low risk", "Moderate risk", "High risk", "Very high risk"], fill_value=0)
                .plot(kind="bar", ax=ax)
            )
            ax.set_xlabel("Risk category")
            ax.set_ylabel("Count")
            ax.set_title("Risk Category Distribution")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            df["bp_status"].value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel("BP class")
            ax.set_ylabel("Count")
            ax.set_title("Blood Pressure Classification")
            st.pyplot(fig)

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df["risk_total_score"], bins=10)
            ax.set_xlabel("Risk score")
            ax.set_ylabel("Frequency")
            ax.set_title("Risk Score Distribution")
            st.pyplot(fig)

        with col4:
            comm = df.groupby("community", dropna=False)["measured_hypertension"].mean().reset_index()
            comm["hypertension_prevalence_percent"] = (comm["measured_hypertension"] * 100).round(2)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(comm["community"].astype(str), comm["hypertension_prevalence_percent"])
            ax.set_xlabel("Community")
            ax.set_ylabel("Prevalence (%)")
            ax.set_title("Community-level Hypertension Prevalence")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        st.dataframe(df.head(50), use_container_width=True)

with tabs[2]:
    st.subheader("Community Summary")
    df = apply_filters(load_data())
    if df.empty:
        st.info("No data available for the selected filters.")
    else:
        summary = df.groupby("community", dropna=False).agg(
            participants=("id", "count"),
            mean_age=("age", "mean"),
            mean_bmi=("bmi", "mean"),
            mean_sbp=("mean_sbp", "mean"),
            mean_dbp=("mean_dbp", "mean"),
            hypertension_prevalence=("measured_hypertension", "mean"),
            diabetes_prevalence=("known_diabetes", "mean"),
            tobacco_prevalence=("tobacco_use", "mean"),
            obesity_prevalence=("bmi", lambda x: np.mean(x >= 30)),
            high_risk_prevalence=("risk_category", lambda x: np.mean(x.isin(["High risk", "Very high risk"])))
        ).reset_index()

        for col in ["mean_age", "mean_bmi", "mean_sbp", "mean_dbp"]:
            summary[col] = summary[col].round(2)

        for col in ["hypertension_prevalence", "diabetes_prevalence", "tobacco_prevalence", "obesity_prevalence", "high_risk_prevalence"]:
            summary[col] = (summary[col] * 100).round(2)

        st.dataframe(summary, use_container_width=True)

with tabs[3]:
    st.subheader("Records Manager")
    df = apply_filters(load_data())
    if df.empty:
        st.info("No records found.")
    else:
        view_cols = [
            "id", "study_id", "visit_type", "parent_id", "visit_date",
            "community", "sex", "age", "mean_sbp", "mean_dbp",
            "risk_total_score", "risk_category"
        ]
        st.dataframe(df[view_cols], use_container_width=True)
        selected_record_id = st.selectbox("Select record ID", df["id"].tolist())
        selected_row = load_record_by_id(int(selected_record_id))
        st.dataframe(selected_row, use_container_width=True)

        if st.button("Delete this record"):
            delete_record(int(selected_record_id))
            st.success(f"Deleted record {selected_record_id}.")
            st.rerun()

with tabs[4]:
    st.subheader("Data Export")
    df = apply_filters(load_data())
    if df.empty:
        st.info("No data available for export.")
    else:
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "lr_hindri_dataset.csv",
            "text/csv"
        )

        community_summary = df.groupby("community", dropna=False).agg(
            participants=("id", "count"),
            mean_age=("age", "mean"),
            mean_bmi=("bmi", "mean"),
            mean_sbp=("mean_sbp", "mean"),
            mean_dbp=("mean_dbp", "mean"),
            hypertension_prevalence=("measured_hypertension", "mean"),
        ).reset_index()
        community_summary["hypertension_prevalence"] = (
            community_summary["hypertension_prevalence"] * 100
        ).round(2)

        excel_bytes = df_to_excel_bytes({
            "raw_data": df,
            "community_summary": community_summary
        })

        st.download_button(
            "Download Excel workbook",
            excel_bytes,
            "lr_hindri_export.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with tabs[5]:
    st.subheader("Machine Learning Calibration")
    st.caption("Pilot logistic regression to predict measured hypertension.")
    df = apply_filters(load_data())

    if df.empty or len(df) < 30:
        st.warning("At least 30 filtered records are recommended.")
    else:
        candidate_features = [
            "age", "bmi", "waist_cm", "family_history_htn", "tobacco_use",
            "known_diabetes", "known_kidney_disease", "known_cvd",
            "missed_bp_meds", "frequent_stress", "missed_care_transport",
            "medicine_runout", "risk_total_score",
        ]
        model_df = df[candidate_features + ["measured_hypertension"]].dropna().copy()
        X = model_df[candidate_features]
        y = model_df["measured_hypertension"]

        if len(model_df) < 30 or y.nunique() < 2:
            st.warning("Not enough outcome variation for model training.")
        else:
            test_size = st.slider("Test size", 0.2, 0.4, 0.3, 0.05)
            random_state = st.number_input("Random state", 1, 9999, 42)

            if st.button("Run calibration model"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                model = LogisticRegression(max_iter=2000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                st.metric("ROC-AUC", round(roc_auc_score(y_test, y_prob), 3))
                st.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))

                cm_df = pd.DataFrame(
                    confusion_matrix(y_test, y_pred),
                    index=["Actual 0", "Actual 1"],
                    columns=["Predicted 0", "Predicted 1"]
                )
                st.markdown("#### Confusion Matrix")
                st.dataframe(cm_df, use_container_width=True)

                coef_df = pd.DataFrame({
                    "feature": candidate_features,
                    "coefficient": model.coef_[0]
                }).sort_values("coefficient", ascending=False)
                st.markdown("#### Model Coefficients")
                st.dataframe(coef_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(coef_df["feature"], coef_df["coefficient"])
                ax.set_xlabel("Coefficient")
                ax.set_title("Logistic Regression Feature Effects")
                st.pyplot(fig)

                report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                st.markdown("#### Classification Report")
                st.dataframe(report_df, use_container_width=True)

with tabs[6]:
    st.subheader("Admin")
    if not st.session_state.admin_logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.success("Logged in successfully.")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    else:
        st.success("Admin logged in.")
        df = load_data()
        st.write(f"Total records in database: **{len(df)}**")

        if st.button("Log out"):
            st.session_state.admin_logged_in = False
            st.rerun()

        st.warning("This action will permanently delete all records.")
        confirm_delete = st.checkbox("I understand and want to delete all records")
        if st.button("Delete all data"):
            if confirm_delete:
                delete_all_data()
                st.success("All data deleted.")
                st.rerun()
            else:
                st.error("Please confirm deletion first.")
