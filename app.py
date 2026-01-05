from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

def is_image_blurry(image_bytes, threshold=200.0):
    # Convert bytes → numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode as image
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True, 0.0

    # Laplacian variance
    lap = cv2.Laplacian(img, cv2.CV_64F)
    variance = float(lap.var())   # <--- CONVERT TO PYTHON FLOAT

    # Blurry if variance < threshold
    blurry = bool(variance < threshold)   # <--- CONVERT TO PYTHON BOOL
    print("Received bytes:", len(image_bytes))
    return blurry, variance

@app.route("/check-blur", methods=["POST"])
def check_blur():
    try:
        image_bytes = request.data

        if not image_bytes:
            return jsonify({"error": "No image data received"}), 400

        blurry, score = is_image_blurry(image_bytes)

        # Convert everything to native Python types
        response = {
            "isBlurry": bool(blurry),
            "score": float(score)
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="localhost", port=5000)
