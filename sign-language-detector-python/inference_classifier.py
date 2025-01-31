import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the trained model and label encoder
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
label_encoder = model_dict['label_encoder']  # Load the label encoder

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize variables to store the current word and sequence
current_word = ""
word_list = []

# Variables to manage the gesture detection
last_predicted_character = None
last_prediction_time = 0
cooldown_threshold = 1  # Cooldown period in seconds before allowing another gesture
last_delete_time = 0  # Track last delete gesture time
delete_cooldown = 0.5  # Cooldown period for delete gesture (shorter for fast deletion)
gesture_timeout = 1.5  # Time to wait before recognizing a new word after gesture (to allow hand removal)
typing_delay = 0.5  # Time delay between typing each character (in seconds)

# Time when last word was completed (space or delete gesture)
word_break_time = 0
space_pause_time = 0  # Variable to handle 3-second pause after space

while True:
    data_aux = []  # List to store the features
    x_ = []  # Temporary list to store x coordinates
    y_ = []  # Temporary list to store y coordinates

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    H, W, _ = frame.shape

    # Convert the frame to RGB (MediaPipe requires RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to detect hand landmarks
    results = hands.process(frame_rgb)

    current_time = time.time()

    # If enough time has passed since typing the last word or character, allow new typing
    if (current_time - word_break_time) > typing_delay and (current_time - space_pause_time) > 3:
        typing_paused = False
    else:
        typing_paused = True  # Pause typing but continue gesture capture

    # If hands are detected, process each hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Reset x_ and y_ for each hand (ensure we handle multiple hands)
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

        # Handle single hand (add padding to make the features 84)
        if len(results.multi_hand_landmarks) == 1:
            print("Single hand detected. Padding the features.")
            # Duplicate the features for the second hand (padding)
            data_aux.extend(data_aux)  # Copy the features of the first hand

        # Ensure that we have 84 features (42 features per hand)
        if len(data_aux) == 84:  # Now we expect 84 features for two hands
            print(f"Number of features for prediction: {len(data_aux)}")
            
            # Prediction if we have the correct number of features
            prediction = model.predict([np.asarray(data_aux)])

            # Decode the predicted label using the label encoder
            predicted_character = label_encoder.inverse_transform(prediction)[0]

            # Only allow new predictions after the cooldown period
            if predicted_character != last_predicted_character or (current_time - last_prediction_time) > cooldown_threshold:
                # Check if the gesture has changed or enough time has passed
                if predicted_character != 'space' and predicted_character != 'del' and not typing_paused:
                    # Append the character to the word if it's valid (not space or delete)
                    current_word += predicted_character
                    print(f"Current word: {current_word}")  # Debugging output

                    # Update last prediction details
                    last_predicted_character = predicted_character
                    last_prediction_time = current_time

                    # After typing a character, pause for typing delay (typing delay between characters)
                    word_break_time = current_time  # Mark the time when we typed a character

                # Handle space and delete gestures
                if predicted_character == 'space':  # Add a space to the word
                    if current_word:  # Only add if there is a word to add
                        word_list.append(current_word)  # Save current word
                        current_word = ""  # Start new word
                    print(f"Word added: {current_word}")  # Debugging output
                    space_pause_time = current_time  # Pause for 3 seconds after space
                    word_break_time = current_time  # Mark time of space gesture

                elif predicted_character == 'del':  # Delete the last character
                    # Allow delete gesture immediately, even if the hand is still visible
                    if (current_time - last_delete_time) > delete_cooldown:
                        if current_word:  # Only delete if the current word isn't empty
                            current_word = current_word[:-1]
                            print(f"Word after delete: {current_word}")  # Debugging output
                        last_delete_time = current_time  # Update last delete time
                        word_break_time = current_time  # Mark time after delete gesture

            # Reset the current word after a successful delete, ensuring it accepts new words
            if (current_time - last_delete_time) > delete_cooldown:
                last_delete_time = current_time  # Ensure the delete cooldown is respected

            # Draw bounding box and label on the frame if prediction is made
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the current word or full sentence
    full_sentence = ' '.join(word_list) + (' ' + current_word if current_word else '')
    cv2.putText(frame, "Word: " + full_sentence, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame with the prediction
    cv2.imshow('frame', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
