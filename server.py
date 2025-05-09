import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from callmodel import get_recommendations

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route("/move", methods=["POST"])
def move():
    data = request.get_json() or {}
    lat = data.get("latitude")
    lon = data.get("longitude")
    mood = data.get("mood")
    if lat is None or lon is None or not mood:
        return jsonify({"error": "latitude, longitude, and mood are required"}), 400

    recs = get_recommendations(user_mood=mood, latitude=lat, longitude=lon)
    return jsonify({"recommendations": recs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
