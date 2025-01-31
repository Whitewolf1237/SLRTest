import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the class labels as alphabetic values or any other labels you use for ASL
class_labels = ['A', 'B', 'C', 'D', 'E', 'F']  # Adjust this based on your dataset

dataset_size = 100  # Number of images to collect for each class

# Open the webcam
cap = cv2.VideoCapture(0)

# Create directories for each class based on the label names
for index, label in enumerate(class_labels):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class: {label}')

    done = False
    while not done:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Wait for 'Q' to start collecting images for the current label
        if cv2.waitKey(25) & 0xFF == ord('q'):
            done = True

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        
        # Display instructions using the alphabetic label for the class
        cv2.putText(frame, f'Collecting image of {label} ({counter + 1}/{dataset_size})', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.waitKey(25)
        
        # Save the collected image with the class label as the folder name
        image_path = os.path.join(DATA_DIR, label, '{}.jpg'.format(counter))
        cv2.imwrite(image_path, frame)
        counter += 1

    print(f"Finished collecting data for class {label}")

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
