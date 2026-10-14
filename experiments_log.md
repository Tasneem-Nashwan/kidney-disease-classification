# 🧪 Experiments Log

Detailed record of all training experiments conducted.

## Summary Table

| # | Model | Technique | Accuracy | Stone Recall | Tumor Recall | Status |
|---|---|---|---|---|---|---|
| v1 | Simple CNN | Baseline | 85.58% | 0.5970 | 0.7055 | Baseline |
| v2 | EfficientNetB0 | Transfer Learning + class_weight | **94.42%** | **0.8859** | **0.9018** | ✅ **Best** |
| v3 | EfficientNetB0 | Strong Augmentation | 87.55% | 0.8365 | 0.6994 | Rejected |
| v4 | EfficientNetB0 | Intensity Augmentation | 78.51% | 0.8403 | 0.3722 | Rejected |

## Detailed Configuration

### v1 — Baseline CNN
- **Architecture**: Custom CNN (3 Conv layers + Dense)
- **Epochs**: 10
- **Class Weight**: ❌
- **Result**: 85.58%
- **Issue**: Poor Stone/Tumor recall due to class imbalance

### v2 — EfficientNetB0 (Best) ⭐
- **Architecture**: EfficientNetB0 (ImageNet pretrained)
- **Epochs**: 13 (with EarlyStopping)
- **Class Weight**: ✅
- **Augmentation**: Moderate
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Result**: **94.42%**
- **Improvements**: +28% Stone recall, +19% Tumor recall vs v1

### v3 — Strong Augmentation
- **Epochs**: 18
- **Augmentation**: Strong (vertical flip, rotation 0.15, translation, contrast, brightness)
- **Result**: 87.55%
- **Conclusion**: Too much augmentation hurt performance

### v4 — Intensity Augmentation
- **Epochs**: 20
- **Augmentation**: Very strong (brightness 0.3, contrast 0.3, gamma)
- **Result**: 78.51%
- **Conclusion**: Failed — model became "confused", Tumor recall dropped to 37%

## Key Learnings

1. **Class Weight is essential** for imbalanced medical datasets
2. **Transfer Learning** with EfficientNetB0 provides excellent results
3. **Moderate augmentation** is better than aggressive augmentation
4. **Domain Shift** remains a challenge for external images

## Final Production Model

- **Model**: v2 (EfficientNetB0 + class_weight)
- **Accuracy**: 94.42%
- **Inference**: Ensemble + TTA (10 augmentations) + CLAHE preprocessing
