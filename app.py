from flask import Flask, request, jsonify, render_template_string
import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from model import DeepfakeModel

app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.join(os.path.dirname(BASE_DIR), "website")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the deepfake model
model = DeepfakeModel()
model_path = os.path.join(BASE_DIR, "deepfake_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Serve the HTML (embed it directly or load from file)
@app.route('/')
def home():
    # Read and serve aidriven.html
    html_path = os.path.join(WEBSITE_DIR, "aidriven.html")
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return jsonify({'error': f'HTML file not found at {html_path}'}), 500
    except UnicodeDecodeError:
        return jsonify({'error': 'Unable to read HTML file due to encoding issues. Please save aidriven.html as UTF-8.'}), 500

# Endpoint for file analysis (handles both images and videos)
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Determine file type
    file_ext = os.path.splitext(file.filename)[1].lower()
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    
    # Save uploaded file temporarily
    temp_path = os.path.join(BASE_DIR, 'temp_upload' + file_ext)
    file.save(temp_path)
    
    try:
        if file_ext in image_extensions:
            return analyze_image(temp_path)
        elif file_ext in video_extensions:
            return analyze_video(temp_path)
        else:
            return jsonify({'error': 'Unsupported file format. Use image or video files.'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def analyze_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Transform for model
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    
    label = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob
    
    # Detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    num_faces = len(faces)
    
    result = {
        'prediction': label,
        'confidence': round(confidence * 100, 2),
        'faces_detected': num_faces
    }
    
    return jsonify(result)

def analyze_video(video_path):
    """Extract frames from video and run predictions on sample frames"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return jsonify({'error': 'Could not read video file'}), 400
    
    frame_count = 0
    predictions = []
    confidence_scores = []
    
    # Process every 10th frame to speed up analysis
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 10th frame
        if frame_count % 10 == 0:
            try:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Transform for model
                img_tensor = transform(frame_rgb).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.sigmoid(output).item()
                
                confidence = prob if prob > 0.5 else 1 - prob
                label = "FAKE" if prob > 0.5 else "REAL"
                
                predictions.append(label)
                confidence_scores.append(confidence)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue
    
    cap.release()
    
    if not predictions:
        return jsonify({'error': 'Could not process video frames'}), 400
    
    # Aggregate results
    fake_count = predictions.count("FAKE")
    real_count = predictions.count("REAL")
    
    # Determine overall prediction based on majority
    overall_prediction = "FAKE" if fake_count > real_count else "REAL"
    avg_confidence = np.mean(confidence_scores)
    
    result = {
        'prediction': overall_prediction,
        'confidence': round(avg_confidence * 100, 2),
        'frames_analyzed': len(predictions),
        'fake_frames': fake_count,
        'real_frames': real_count
    }
    
    return jsonify(result)

# New endpoint for real-time prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Read the image from file
    img_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Transform
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    
    label = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob
    
    return jsonify({
        'prediction': label,
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)