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
model = tf.keras.models.load_model(model_path)

print("Loaded model:", model_path)
print("Model input shape:", model.input_shape)
print("Class mapping:", class_indices)


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
    # Determine target size from model if possible
    target = None
    try:
        ishape = model.input_shape
        if ishape is not None and len(ishape) >= 3 and ishape[1] is not None and ishape[2] is not None:
            target = (ishape[1], ishape[2])
    except Exception:
        target = (224, 224)

    x = preprocess_image(img_path, target_size=target)
    preds = model.predict(x)
    # Normalize logits to probabilities
    if preds.ndim == 2:
        raw = preds[0]
    else:
        raw = preds
    probs = tf.nn.softmax(raw).numpy()
    idx = int(np.argmax(probs))
    return idx, probs


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_predictions.py <image1> [image2 ...]")
        sys.exit(1)

    for p in sys.argv[1:]:
        if not os.path.exists(p):
            print(p, "-> file not found")
            continue
        idx, probs = predict(p)
        class_name = class_indices.get(str(idx), f"index_{idx}")
        print(f"{p} -> predicted: {class_name} (index {idx})")
        print("probabilities:")
        for i, prob in enumerate(probs):
            print(f"  {i}: {class_indices.get(str(i),'?')} = {prob:.4f}")
        print("---")
