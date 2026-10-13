from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')

PIMA = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
RCOLS = ['race', 'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 'num_medications',
         'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
         'A1Cresult', 'insulin', 'change', 'diabetesMed']

# name: (UCI id, target column or None for first column, target function)
DISEASES = {
    'Heart': (45, None, lambda s: (s > 0).astype(int)),
    'Kidney': (336, None, lambda s: (s.str.strip() == 'ckd').astype(int)),
    'Liver': (225, None, lambda s: (s == 1).astype(int)),
    'Cervical Cancer': (383, 'Biopsy', lambda s: s.astype(int)),
    'Parkinson': (174, None, lambda s: s.astype(int)),
    'Hepatitis': (46, None, lambda s: (s == 1).astype(int)),
    'Diabetes': None,
}

LABELS = {
    'Heart': {
        'age': 'Age (years)',
        'sex': 'Sex (1 = Male, 0 = Female)',
        'cp': 'Chest Pain Type (0 to 3)',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Serum Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)',
        'restecg': 'Resting ECG Result (0 to 2)',
        'thalach': 'Maximum Heart Rate Achieved',
        'exang': 'Exercise Induced Angina (1 = Yes, 0 = No)',
        'oldpeak': 'ST Depression Induced by Exercise',
        'slope': 'Slope of Peak Exercise ST Segment (1 to 3)',
        'ca': 'Number of Major Vessels Colored by Fluoroscopy (0 to 3)',
        'thal': 'Thalassemia (3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect)',
    },
    'Kidney': {
        'age': 'Age (years)',
        'bp': 'Blood Pressure (mm Hg)',
        'sg': 'Specific Gravity of Urine',
        'al': 'Albumin Level in Urine',
        'su': 'Sugar Level in Urine',
        'rbc': 'Red Blood Cells in Urine',
        'pc': 'Pus Cell in Urine',
        'pcc': 'Pus Cell Clumps',
        'ba': 'Bacteria Present',
        'bgr': 'Blood Glucose Random (mg/dl)',
        'bu': 'Blood Urea (mg/dl)',
        'sc': 'Serum Creatinine (mg/dl)',
        'sod': 'Sodium Level (mEq/L)',
        'pot': 'Potassium Level (mEq/L)',
        'hemo': 'Hemoglobin (gms)',
        'pcv': 'Packed Cell Volume',
        'wbcc': 'White Blood Cell Count (cells/cumm)',
        'rbcc': 'Red Blood Cell Count (millions/cmm)',
        'htn': 'Hypertension (yes / no)',
        'dm': 'Diabetes Mellitus (yes / no)',
        'cad': 'Coronary Artery Disease (yes / no)',
        'appet': 'Appetite (good / poor)',
        'pe': 'Pedal Edema (yes / no)',
        'ane': 'Anemia (yes / no)',
    },
    'Liver': {
        'Age': 'Age (years)',
        'Gender': 'Gender',
        'TB': 'Total Bilirubin (mg/dl)',
        'DB': 'Direct Bilirubin (mg/dl)',
        'Alkphos': 'Alkaline Phosphotase (IU/L)',
        'Sgpt': 'Alamine Aminotransferase (IU/L)',
        'Sgot': 'Aspartate Aminotransferase (IU/L)',
        'TP': 'Total Proteins (g/dl)',
        'ALB': 'Albumin (g/dl)',
        'A/G Ratio': 'Albumin and Globulin Ratio',
    },
    'Cervical Cancer': {
        'Age': 'Age (years)',
        'Number of sexual partners': 'Number of Sexual Partners',
        'First sexual intercourse': 'Age at First Sexual Intercourse (years)',
        'Num of pregnancies': 'Number of Pregnancies',
        'Smokes': 'Smokes (1 = Yes, 0 = No)',
        'Smokes (years)': 'Years of Smoking',
        'Smokes (packs/year)': 'Packs of Cigarettes per Year',
        'Hormonal Contraceptives': 'Uses Hormonal Contraceptives (1 = Yes, 0 = No)',
        'Hormonal Contraceptives (years)': 'Years of Hormonal Contraceptive Use',
        'IUD': 'Uses IUD (1 = Yes, 0 = No)',
        'IUD (years)': 'Years of IUD Use',
        'STDs': 'Has Had Any Sexually Transmitted Disease (1 = Yes, 0 = No)',
        'STDs (number)': 'Number of STDs',
        'STDs:condylomatosis': 'STD: Condylomatosis',
        'STDs:cervical condylomatosis': 'STD: Cervical Condylomatosis',
        'STDs:vaginal condylomatosis': 'STD: Vaginal Condylomatosis',
        'STDs:vulvo-perineal condylomatosis': 'STD: Vulvo-perineal Condylomatosis',
        'STDs:syphilis': 'STD: Syphilis',
        'STDs:pelvic inflammatory disease': 'STD: Pelvic Inflammatory Disease',
        'STDs:genital herpes': 'STD: Genital Herpes',
        'STDs:molluscum contagiosum': 'STD: Molluscum Contagiosum',
        'STDs:AIDS': 'STD: AIDS',
        'STDs:HIV': 'STD: HIV',
        'STDs:Hepatitis B': 'STD: Hepatitis B',
        'STDs:HPV': 'STD: HPV',
        'STDs: Number of diagnosis': 'Number of STD Diagnoses',
        'STDs: Time since first diagnosis': 'Time Since First STD Diagnosis (years)',
        'STDs: Time since last diagnosis': 'Time Since Last STD Diagnosis (years)',
        'Dx:Cancer': 'Diagnosed with Cancer (1 = Yes, 0 = No)',
        'Dx:CIN': 'Diagnosed with CIN (Cervical Intraepithelial Neoplasia)',
        'Dx:HPV': 'Diagnosed with HPV',
        'Dx': 'Diagnosed with Any Condition',
    },
    'Parkinson': {
        'MDVP:Fo(Hz)': 'Average Vocal Fundamental Frequency (Hz)',
        'MDVP:Fhi(Hz)': 'Maximum Vocal Fundamental Frequency (Hz)',
        'MDVP:Flo(Hz)': 'Minimum Vocal Fundamental Frequency (Hz)',
        'MDVP:Jitter(%)': 'Jitter Percentage',
        'MDVP:Jitter(Abs)': 'Absolute Jitter',
        'MDVP:RAP': 'Relative Amplitude Perturbation',
        'MDVP:PPQ': 'Five-point Period Perturbation Quotient',
        'Jitter:DDP': 'Average Absolute Difference of Differences of Jitter',
        'MDVP:Shimmer': 'Shimmer',
        'MDVP:Shimmer(dB)': 'Shimmer in Decibels',
        'Shimmer:APQ3': 'Three-point Amplitude Perturbation Quotient',
        'Shimmer:APQ5': 'Five-point Amplitude Perturbation Quotient',
        'MDVP:APQ': 'Amplitude Perturbation Quotient',
        'Shimmer:DDA': 'Average Absolute Differences of Shimmer',
        'NHR': 'Noise to Harmonic Ratio',
        'HNR': 'Harmonic to Noise Ratio',
        'RPDE': 'Recurrence Period Density Entropy',
        'DFA': 'Detrended Fluctuation Analysis',
        'spread1': 'Nonlinear Frequency Spread 1',
        'spread2': 'Nonlinear Frequency Spread 2',
        'D2': 'Correlation Dimension',
        'PPE': 'Pitch Period Entropy',
    },
    'Hepatitis': {
        'age': 'Age (years)',
        'sex': 'Sex (1 = Male, 2 = Female)',
        'steroid': 'Takes Steroids (1 = Yes, 2 = No)',
        'antivirals': 'Takes Antivirals (1 = Yes, 2 = No)',
        'fatigue': 'Fatigue (1 = Yes, 2 = No)',
        'malaise': 'Malaise / General Discomfort (1 = Yes, 2 = No)',
        'anorexia': 'Loss of Appetite (1 = Yes, 2 = No)',
        'liver_big': 'Enlarged Liver (1 = Yes, 2 = No)',
        'liver_firm': 'Firm Liver (1 = Yes, 2 = No)',
        'spleen_palpable': 'Enlarged Spleen (1 = Yes, 2 = No)',
        'spiders': 'Spider Angiomas (1 = Yes, 2 = No)',
        'ascites': 'Fluid in Abdomen / Ascites (1 = Yes, 2 = No)',
        'varices': 'Enlarged Veins / Varices (1 = Yes, 2 = No)',
        'bilirubin': 'Bilirubin Level (mg/dl)',
        'alk_phosphate': 'Alkaline Phosphatase (U/L)',
        'sgot': 'SGOT Enzyme (U/L)',
        'albumin': 'Albumin Level (g/dl)',
        'protime': 'Prothrombin Time (seconds)',
        'histology': 'Liver Biopsy Histology (1 = Yes, 2 = No)',
    },
    'Diabetes': {
        'Pregnancies': 'Number of Pregnancies',
        'Glucose': 'Plasma Glucose Concentration (mg/dl)',
        'BloodPressure': 'Diastolic Blood Pressure (mm Hg)',
        'SkinThickness': 'Triceps Skin Fold Thickness (mm)',
        'Insulin': 'Serum Insulin (mu U/ml)',
        'BMI': 'Body Mass Index',
        'DiabetesPedigree': 'Diabetes Pedigree Function (family history score)',
        'Age': 'Age (years)',
    },
}

RAW = {}
CACHE = {}


def clean(df):
    df = df.replace('?', np.nan)
    for c in df.columns:
        n = pd.to_numeric(df[c], errors='coerce')
        if n.notna().sum() >= df[c].notna().sum():
            df[c] = n
        else:
            df[c] = df[c].where(df[c].isna(), df[c].astype(str))
    return df


def load_raw(key):
    if key in RAW:
        return RAW[key]
    if key == 'readmit':
        d = fetch_ucirepo(id=296)
        raw = clean(d.data.features[RCOLS].copy())
        raw['age'] = raw['age'].astype(str).str.extract(r'(\d+)')[0].astype(float)
        y = (d.data.targets.iloc[:, 0] == '<30').astype(int)
    elif key == 'Diabetes':
        df = pd.read_csv('pima-indians-diabetes.csv', header=None, names=PIMA)
        raw = df.drop(columns='Outcome').replace(0, np.nan)
        y = df['Outcome']
    else:
        id_, col, fix = DISEASES[key]
        d = fetch_ucirepo(id=id_)
        if d.data.targets is not None:
            raw = clean(d.data.features).dropna(axis=1, how='all')
            target = d.data.targets[col] if col else d.data.targets.iloc[:, 0]
        else:
            full = d.data.original
            target = full[col]
            raw = clean(full.drop(columns=[col])).dropna(axis=1, how='all')
        mask = target.notna().values
        raw = raw[mask].reset_index(drop=True)
        y = fix(target[mask].reset_index(drop=True))
    RAW[key] = (raw, y)
    return RAW[key]


def get_model(key):
    if key not in CACHE:
        raw, y = load_raw(key)
        Xe = pd.get_dummies(raw).astype(float)
        med = Xe.median()
        if key == 'readmit':
            model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                                  eval_metric='logloss')
        else:
            model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
        model.fit(Xe.fillna(med), y)
        CACHE[key] = {'model': model, 'cols': list(Xe.columns), 'med': med}
    return CACHE[key]


@app.get('/')
def home():
    return FileResponse('static/index.html')


@app.get('/api/diseases')
def diseases():
    return list(DISEASES)


@app.get('/api/fields/{key}')
def fields(key: str):
    if key != 'readmit' and key not in DISEASES:
        raise HTTPException(404, 'Unknown disease')
    try:
        raw, _ = load_raw(key)
    except Exception as e:
        raise HTTPException(500, f'Data load error: {e}')
    labels = LABELS.get(key, {})
    out = []
    for c in raw.columns:
        s = raw[c]
        label = labels.get(c, c.replace('_', ' '))
        if pd.api.types.is_numeric_dtype(s):
            med = s.median()
            out.append({'name': c, 'label': label, 'type': 'number',
                        'default': None if pd.isna(med) else round(float(med), 2)})
        else:
            opts = sorted({str(v) for v in s.dropna().unique()})
            out.append({'name': c, 'label': label, 'type': 'select', 'options': opts})
    return out


@app.post('/api/predict/{key}')
def predict(key: str, body: dict = Body(...)):
    if key != 'readmit' and key not in DISEASES:
        raise HTTPException(404, 'Unknown disease')
    try:
        b = get_model(key)
        raw, _ = load_raw(key)
        row = pd.DataFrame([body])
        for c in raw.columns:
            if not pd.api.types.is_numeric_dtype(raw[c]) and c in row and row[c].notna().any():
                row[c] = row[c].apply(lambda v: None if v is None else str(v))
        if key == 'Diabetes':
            row = row.replace(0, np.nan)
        e = pd.get_dummies(row).reindex(columns=b['cols'], fill_value=0).astype(float)
        e = e.fillna(b['med'])
        p = float(b['model'].predict_proba(e)[0, 1])
        return {'probability': round(p * 100, 1)}
    except Exception as e:
        raise HTTPException(500, f'Prediction error: {e}')