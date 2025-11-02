from flask import Flask, render_template, request, send_file
import cv2
import json
import os
import numpy as np

app = Flask(__name__)

# Paths
IMAGE_PATH = "ceiling-photo.jpg"
DETECTIONS_PATH = "plate_detections.json"
OUTPUT_PATH = "static/highlighted.jpg"

os.makedirs("static", exist_ok=True)

@app.route("/")
def index():
    states = sorted([
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'Aruba', 'California', 'Canada', 
        'Cayman Islands', 'Colorado', 'Connecticut', 'Curacao', 'Delaware', 'Florida', 
        'Georgia', 'Grand Cayman', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 
        'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 
        'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 
        'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Panama', 
        'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 
        'Texas', 'US Government', 'Utah', 'Venezuela', 'Vermont', 'Virginia', 
        'Washington', 'Washington D.C.', 'Washington DC', 'West Virginia', 
        'Wisconsin', 'Wyoming'
    ])
    return render_template("index.html", states=states)

@app.route("/get_image")
def get_image():
    state = request.args.get("state")
    if not state:
        return "Error: no state provided", 400

    # Load image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        return "Error: could not load image", 500

    # Create white background same size as original image
    spotlight = 255 * np.ones_like(img)

    # Load detections
    with open(DETECTIONS_PATH) as f:
        detections = json.load(f)

    # Copy only matching plates onto white image
    for d in detections:
        if d["state"].lower() == state.lower():
            x, y, w, h = d["x"], d["y"], d["width"], d["height"]
            spotlight[y:y+h, x:x+w] = img[y:y+h, x:x+w]

    # Save result
    cv2.imwrite(OUTPUT_PATH, spotlight)

    return send_file(OUTPUT_PATH, mimetype="image/jpeg")

if __name__ == "__main__":
    import numpy as np
    app.run(debug=True)