import os
import pickle
import numpy as np
import cv2
from flask import Flask, render_template, Response, request, jsonify
import mediapipe as mp

app = Flask(__name__)

# Load the trained model and label encoder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, 'model.p')

model_dict = pickle.load(open(MODEL_PATH, 'rb'))
model = model_dict['model']
label_encoder = model_dict['label_encoder']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Real-time camera capture route
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        data_aux = []  # List to store features
        x_ = []  # Temporary list to store x coordinates
        y_ = []  # Temporary list to store y coordinates

        H, W, _ = frame.shape

        # Convert the frame to RGB (MediaPipe requires RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Reset x_ and y_ for each hand (handle multiple hands)
                x_ = []  
                y_ = []

                # Extract x and y coordinates for each hand landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize the landmark coordinates (relative to min x and y)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # normalized x
                    data_aux.append(y - min(y_))  # normalized y

            # If only one hand is detected, pad the features
            if len(results.multi_hand_landmarks) == 1:
                print("Single hand detected. Padding the features.")
                data_aux.extend(data_aux)  # Duplicate the features of the first hand

            # Ensure we have 84 features (42 per hand)
            if len(data_aux) == 84:  # Expecting 84 features (42 per hand)
                print(f"Number of features for prediction: {len(data_aux)}")

                # Make prediction using the model
                prediction = model.predict([np.asarray(data_aux)])

                # Decode the predicted label using the label encoder
                predicted_character = label_encoder.inverse_transform(prediction)[0]

                # Draw bounding box and predicted label
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                # Change color of the bounding box and text to green
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Green bounding box
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)  # Green text
            else:
                print("Invalid number of features. Skipping this frame.")

        # Show the frame with prediction
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to upload images and predict
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read and process the uploaded image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    data_aux = []
    x_ = []
    y_ = []

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = label_encoder.inverse_transform(prediction)[0]
            return jsonify({'prediction': predicted_character})

    return jsonify({'error': 'No valid hand landmarks found'}), 400

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
