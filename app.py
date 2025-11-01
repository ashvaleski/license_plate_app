from flask import Flask, render_template, request, send_file
import json
import cv2
import io

app = Flask(__name__)

# Load detections once
with open("plate_detections.json", "r") as f:
    detections = json.load(f)

IMAGE_PATH = "ceiling-photo.jpg"

@app.route("/")
def index():
    # Get all unique states for dropdown
    states = sorted(set(d.get("state", "Unknown") for d in detections))
    return render_template("index.html", states=states)

@app.route("/highlight", methods=["POST"])
def highlight():
    state_query = request.form.get("state", "").lower()
    image = cv2.imread(IMAGE_PATH)

    # Draw rectangles: green for match, gray for others
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        label = det.get("state", "Unknown")
        color = (0, 255, 0) if label.lower() == state_query else (180, 180, 180)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 8)
        cv2.putText(image, label, (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Encode as JPEG
    _, buffer = cv2.imencode(".jpg", image)
    return send_file(io.BytesIO(buffer.tobytes()),
                     mimetype="image/jpeg",
                     as_attachment=False,
                     download_name="annotated.jpg")

if __name__ == "__main__":
    app.run(debug=True)
