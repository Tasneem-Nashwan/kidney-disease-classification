import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.ndimage import rotate
import cv2

# ============ تحميل النموذجين ============
print("⏳ جاري تحميل النموذج v2 (الأدق على Kaggle)...")
model_v2 = tf.keras.models.load_model("kidney_model_clean.keras")
print("✅ تم تحميل v2")

print("⏳ جاري تحميل النموذج v3 (الأكثر مرونة)...")
model_v3 = tf.keras.models.load_model("kidney_model_v3_clean.keras")
print("✅ تم تحميل v3")

class_names = ["Cyst", "Normal", "Stone", "Tumor"]
IMG_SIZE = 224
WEIGHT_V2 = 0.6
WEIGHT_V3 = 0.4

print("🚀 كل النماذج جاهزة!")


# ============ معالجة مسبقة بمعيار CT ============
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
    result = np.stack([equalized] * 3, axis=-1)
    return result.astype('float32')


# ============ TTA ============
def predict_single_model(model, arr):
    predictions = []
    predictions.append(model.predict(np.expand_dims(arr, 0), verbose=0)[0])
    predictions.append(model.predict(np.expand_dims(np.fliplr(arr).copy(), 0), verbose=0)[0])
    predictions.append(model.predict(np.expand_dims(np.flipud(arr).copy(), 0), verbose=0)[0])
    for angle in [-12, -6, 6, 12]:
        rot = rotate(arr, angle, reshape=False, mode='nearest', axes=(0, 1))
        predictions.append(model.predict(np.expand_dims(rot, 0), verbose=0)[0])
    predictions.append(model.predict(np.expand_dims(np.clip(arr * 1.15, 0, 255), 0), verbose=0)[0])
    predictions.append(model.predict(np.expand_dims(np.clip(arr * 0.85, 0, 255), 0), verbose=0)[0])
    mean_val = arr.mean()
    predictions.append(model.predict(np.expand_dims(np.clip((arr - mean_val) * 1.2 + mean_val, 0, 255), 0), verbose=0)[0])
    predictions.append(model.predict(np.expand_dims(np.clip((arr - mean_val) * 0.8 + mean_val, 0, 255), 0), verbose=0)[0])
    return np.mean(predictions, axis=0)


# ============ التنبؤ النهائي (Ensemble) ============
def predict_ensemble(img):
    arr = preprocess_ct_image(img)
    pred_v2 = predict_single_model(model_v2, arr)
    pred_v3 = predict_single_model(model_v3, arr)
    # متوسط مرجّح
    avg = WEIGHT_V2 * pred_v2 + WEIGHT_V3 * pred_v3
    # تطبيع
    if avg.sum() > 0:
        avg = avg / avg.sum()
    return {class_names[i]: float(avg[i]) for i in range(len(class_names))}


# ============ الواجهة ============
demo = gr.Interface(
    fn=predict_ensemble,
    inputs=gr.Image(type="pil", label="ارفع صورة CT للكلى"),
    outputs=gr.Label(num_top_classes=4, label="التصنيف المتوقع"),
    title="🩺 نظام تصنيف أمراض الكلى (Ensemble)",
    description="""**النموذج**: Ensemble (EfficientNetB0 v2 + v3) + TTA + CLAHE
**الدقة على Kaggle**: 94.42%
**المعالجة**: CLAHE + Intensity Normalization + Ensemble""",
    allow_flagging="never"
)

print("🚀 جاري تشغيل الواجهة...")
demo.launch(share=True, debug=False)