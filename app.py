 # app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="🚦 Road Collisions Analysis", layout="wide")

# --- Sidebar ---
st.sidebar.header("⚙️ الإعدادات")
uploaded_file = st.sidebar.file_uploader("ارفع ملف البيانات (CSV)", type="csv")

model_choice = st.sidebar.selectbox(
    "اختر النموذج:",
    ["XGBoost", "LightGBM", "Decision Tree", "CatBoost"]
)

# --- Main App ---
st.title("🚦 Road Collisions Analysis & Prediction")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ تم رفع الملف بنجاح")

    # ====== 1) EDA ======
    st.header("🔎 استكشاف البيانات")
    st.write("عدد الصفوف والأعمدة:", df.shape)
    st.write("أول 5 صفوف:")
    st.dataframe(df.head())

    # توزيع الهدف لو موجود
    if "legacy_collision_severity" in df.columns:
        fig = px.histogram(
            df, x="legacy_collision_severity", color="legacy_collision_severity",
            title="توزيع شدة الحوادث"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ====== 2) Preprocessing ======
    st.header("🛠️ المعالجة المسبقة")
    target_col = "legacy_collision_severity"
    if target_col not in df.columns:
        st.error("⚠️ لا يوجد عمود 'legacy_collision_severity' في البيانات")
        st.stop()

    df_clean = df.dropna(subset=[target_col])
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # Encode categoricals
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Scale numeric cols
    num_cols = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.write("✅ تم تجهيز البيانات للتدريب")

    # ====== 3) Models ======
    st.header("🤖 النماذج")

    if model_choice == "XGBoost":
        model = XGBClassifier(
            n_estimators=300, learning_rate=0.01, max_depth=6,
            random_state=42, eval_metric="mlogloss", use_label_encoder=False
        )
    elif model_choice == "LightGBM":
        model = LGBMClassifier(
            n_estimators=300, learning_rate=0.01, random_state=42
        )
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_choice == "CatBoost":
        model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=8,
                                   loss_function="MultiClass", verbose=0, random_seed=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("🎯 دقة النموذج", f"{acc:.2%}")

    # Classification report
    st.subheader("📋 تقرير التصنيف")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.subheader("📊 مصفوفة الالتباس")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
    st.plotly_chart(fig_cm, use_container_width=True)

else:
    st.info("⬅️ من فضلك ارفع ملف البيانات من القائمة الجانبية")
