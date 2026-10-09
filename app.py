import base64
import io

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from scipy import ndimage

from model import DigitCNN

MEAN, STD = 0.1307, 0.3081

app = Flask(__name__)

# ---------- Load trained model once ----------
model = DigitCNN()
model.load_state_dict(torch.load("digit_cnn.pth", map_location="cpu"))
model.eval()


def preprocess(data_url):
    """Turn a canvas drawing into a MNIST-style 1x1x28x28 tensor."""
    encoded = data_url.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")

    # Put the drawing on a black background, then grayscale
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    img = Image.alpha_composite(bg, img).convert("L")
    arr = np.array(img, dtype=np.float32)

    # 1. Crop to the digit's bounding box
    ys, xs = np.where(arr > 30)
    if len(ys) == 0:
        return None                       # nothing drawn
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 2. Scale so the longer side is 20px (keeps aspect ratio)
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = Image.fromarray(arr.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)

    # 3. Paste into the middle of a 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.float32)
    x0, y0 = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = np.array(digit)

    # 4. Shift so the center of mass is the image center (as in MNIST)
    cy, cx = ndimage.center_of_mass(canvas)
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1)
    canvas = np.clip(canvas, 0, 255)

    # 5. Normalize exactly like training
    canvas = (canvas / 255.0 - MEAN) / STD
    return torch.tensor(canvas, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    tensor = preprocess(data["image"])
    if tensor is None:
        return jsonify({"error": "Draw a digit first"}), 400

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    digit = int(probs.argmax())
    return jsonify({
        "digit": digit,
        "confidence": float(probs[digit]),
        "probabilities": [float(p) for p in probs],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)