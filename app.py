import streamlit as st
st.set_page_config(page_title="Kidney Disease Classifier", page_icon="🩺", layout="wide")

import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.ndimage import rotate
import cv2
import os
import requests
import pandas as pd

# ============ تحميل النموذج من Google Drive ============
MODEL_PATH = "kidney_model_clean.keras"
FILE_ID = "1_glpz6OnqWZt5QkRGLbhzZVmJNU9Yhv4"

def download_from_gdrive(file_id, output):
    """تحميل موثوق من Google Drive باستخدام requests"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)
   
    # معالجة صفحة التأكيد للملفات الكبيرة
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(url, stream=True)
            break
   
    with open(output, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ تحميل النموذج لأول مرة (~30 ثانية)..."):
            download_from_gdrive(FILE_ID, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()
class_names = ["Cyst", "Normal", "Stone", "Tumor"]
IMG_SIZE = 224

# ============ معالجة الصور (CLAHE) ============
def preprocess_ct_image(img):
    img_array = np.array(img.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    equalized = equalized.astype('float32')
    p2, p98 = np.percentile(equalized, (2, 98))
    if p98 - p2 > 0:
        equalized = np.clip((equalized - p2) / (p98 - p2) * 255, 0, 255)
    equalized = cv2.resize(equalized, (IMG_SIZE, IMG_SIZE))
    return np.stack([equalized] * 3, axis=-1).astype('float32')

# ============ التنبؤ مع TTA ============
def predict_with_tta(img):
    arr = preprocess_ct_image(img)
    preds = []
    preds.append(model.predict(np.expand_dims(arr, 0), verbose=0)[0])
    preds.append(model.predict(np.expand_dims(np.fliplr(arr).copy(), 0), verbose=0)[0])
    preds.append(model.predict(np.expand_dims(np.flipud(arr).copy(), 0), verbose=0)[0])
    for angle in [-12, 12]:
        rot = rotate(arr, angle, reshape=False, mode='nearest', axes=(0, 1))
        preds.append(model.predict(np.expand_dims(rot, 0), verbose=0)[0])
    preds.append(model.predict(np.expand_dims(np.clip(arr * 1.15, 0, 255), 0), verbose=0)[0])
    preds.append(model.predict(np.expand_dims(np.clip(arr * 0.85, 0, 255), 0), verbose=0)[0])
    avg = np.mean(preds, axis=0)
    if avg.sum() > 0:
        avg = avg / avg.sum()
    return avg

# ============ الواجهة ============
st.title("🩺 نظام تصنيف أمراض الكلى")
st.markdown("**النموذج**: EfficientNetB0 + TTA + CLAHE | **الدقة على بيانات Kaggle**: 94.42%")
st.markdown("---")

uploaded_file = st.file_uploader("📤 ارفعي صورة CT للكلى", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="الصورة المرفوعة", use_column_width=True)
    with col2:
        with st.spinner("🔍 جاري التحليل..."):
            preds = predict_with_tta(img)
        top_idx = int(np.argmax(preds))
        confidence = preds[top_idx] * 100
        st.success(f"### 🎯 التصنيف: **{class_names[top_idx]}**")
        st.metric("نسبة الثقة", f"{confidence:.2f}%")
        st.markdown("### 📊 الاحتمالات لكل فئة:")
        df = pd.DataFrame({
            'الفئة': class_names,
            'الاحتمالية (%)': (preds * 100).round(2)
        }).sort_values('الاحتمالية (%)', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index('الفئة'))

st.markdown("---")
st.caption("⚠️ هذا النموذج لأغراض تعليمية فقط، وليس بديلاً عن التشخيص الطبي.")
