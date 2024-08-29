from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from process_image import run_inference
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process-image', methods=['POST'])
def process_image():
    room_image = request.files.get('roomImage')
    pattern_image = request.files.get('patternImage')
    color_code = request.form.get('colorCode')

    # Save the images temporarily
    room_image_path = 'temp_room_image.jpg'
    pattern_image_path = 'temp_pattern_image.jpg' if pattern_image else None

    room_image.save(room_image_path)
    if pattern_image:
        pattern_image.save(pattern_image_path)

    # Run inference
    processed_image_path = run_inference(room_image_path, pattern_image_path, color_code)

    # Convert processed image to base64
    with open(processed_image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')

    # Return base64 encoded image
    return jsonify({'processedImage': base64_image})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
