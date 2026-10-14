"""
MLflow Tracking for Kidney Disease Classification
يسجّل تجارب التدريب الأربع مع النتائج
"""
import mlflow
import mlflow.keras
import os

# اسم التجربة الرئيسية
mlflow.set_experiment("Kidney_Disease_Classification")

# ============ التجارب التي أجريتها ============
experiments = [
    {
        "name": "v1_baseline_CNN",
        "model": "Simple CNN",
        "epochs": 10,
        "batch_size": 32,
        "img_size": 224,
        "class_weight": False,
        "accuracy": 0.8558,
        "stone_recall": 0.5970,
        "tumor_recall": 0.7055,
        "notes": "Baseline - بدون معالجة، بدون class_weight"
    },
    {
        "name": "v2_EfficientNet_classweight",
        "model": "EfficientNetB0",
        "epochs": 13,
        "batch_size": 32,
        "img_size": 224,
        "class_weight": True,
        "accuracy": 0.9442,
        "stone_recall": 0.8859,
        "tumor_recall": 0.9018,
        "notes": "الأفضل - Transfer Learning + class_weight"
    },
    {
        "name": "v3_strong_augmentation",
        "model": "EfficientNetB0",
        "epochs": 18,
        "batch_size": 32,
        "img_size": 224,
        "class_weight": True,
        "accuracy": 0.8755,
        "stone_recall": 0.8365,
        "tumor_recall": 0.6994,
        "notes": "Augmentation قوي - أضرّ بالأداء"
    },
    {
        "name": "v4_intensity_augmentation",
        "model": "EfficientNetB0",
        "epochs": 20,
        "batch_size": 32,
        "img_size": 224,
        "class_weight": True,
        "accuracy": 0.7851,
        "stone_recall": 0.8403,
        "tumor_recall": 0.3722,
        "notes": "Augmentation قوي جداً - فشل"
    },
]

# ============ تسجيل كل تجربة في MLflow ============
for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):
        # تسجيل المعاملات
        mlflow.log_param("model_architecture", exp["model"])
        mlflow.log_param("epochs", exp["epochs"])
        mlflow.log_param("batch_size", exp["batch_size"])
        mlflow.log_param("image_size", exp["img_size"])
        mlflow.log_param("class_weight_used", exp["class_weight"])
        mlflow.log_param("framework", "TensorFlow 2.16.2")
        mlflow.log_param("pretrained_weights", "ImageNet")

        # تسجيل المقاييس
        mlflow.log_metric("accuracy", exp["accuracy"])
        mlflow.log_metric("stone_recall", exp["stone_recall"])
        mlflow.log_metric("tumor_recall", exp["tumor_recall"])

        # تسجيل الملاحظات
        mlflow.set_tag("notes", exp["notes"])

        print(f"✅ تم تسجيل: {exp['name']} - Accuracy: {exp['accuracy']}")

# ============ تسجيل ملفات إضافية ============
if os.path.exists("confusion_matrix.png"):
    with mlflow.start_run(run_name="v2_artifacts"):
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("training_curves.png")
        print("✅ تم تسجيل الصور")

print("\n🎉 اكتمل التسجيل في MLflow!")
print("📁 موقع البيانات: mlruns/")