import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "corn-or-maize-leaf-disease-dataset.h5")
class_file = os.path.join(working_dir, "class_indices.json")

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    sys.exit(1)

if not os.path.exists(class_file):
    print(f"class_indices.json not found at {class_file}")
    sys.exit(1)

class_indices = json.load(open(class_file))
idx_to_class = {int(k): v for k, v in class_indices.items()}
class_to_idx = {v: k for k, v in idx_to_class.items()}
model = tf.keras.models.load_model(model_path)


def preprocess_image(img_path, target_size=None):
    img = Image.open(img_path).convert('RGB')
    if target_size is not None:
        img = img.resize(target_size)
    else:
        img = img.resize((224, 224))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(img_path):
    target = None
    try:
        ishape = model.input_shape
        if ishape is not None and len(ishape) >= 3 and ishape[1] is not None and ishape[2] is not None:
            target = (ishape[1], ishape[2])
    except Exception:
        target = (224, 224)

    x = preprocess_image(img_path, target_size=target)
    preds = model.predict(x)
    if preds.ndim == 2:
        raw = preds[0]
    else:
        raw = preds
    probs = tf.nn.softmax(raw).numpy()
    idx = int(np.argmax(probs))
    return idx, probs


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <labeled_root_dir>")
        print("Directory format: <root>/<class_name>/*.jpg")
        sys.exit(1)

    root = sys.argv[1]
    y_true = []
    y_pred = []

    # Walk class subfolders
    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        expected_idx = class_to_idx.get(class_name)
        if expected_idx is None:
            print(f"Warning: class '{class_name}' not in class_indices.json; skipping")
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            path = os.path.join(class_dir, fname)
            try:
                idx, probs = predict(path)
                y_true.append(expected_idx)
                y_pred.append(idx)
            except Exception as e:
                print(f"Error processing {path}: {e}")

    # Summarize
    if len(y_true) == 0:
        print("No images found to evaluate.")
        sys.exit(1)

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("Classification report:\n")
        print(classification_report(y_true, y_pred, target_names=[idx_to_class[i] for i in sorted(idx_to_class.keys())]))
        print("Confusion matrix:\n")
        print(confusion_matrix(y_true, y_pred))
    except Exception:
        # Fallback: basic counts
        print("scikit-learn not available; showing simple accuracy and per-class counts")
        total = len(y_true)
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
        classes = sorted(idx_to_class.keys())
        for c in classes:
            tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
            total_c = sum(1 for a in y_true if a == c)
            print(f"Class {idx_to_class[c]}: {tp}/{total_c} correct")
