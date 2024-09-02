#BEST CHOICE AMONGST ALL!!! This is the latest code... It Speaks 'Object-Detected' Whenever a new object is detected or the count of the existing object increases..(COUNT ISSUE!)

import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
import torch
import pyttsx3
from collections import Counter
import threading
import time

# Load YOLOv8 model with GPU support if available
MODEL = "yolov8x.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL).to(device)
model.fuse()

# Dict mapping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# List of class IDs to track in the video stream
selected_classes = [0, 2, 3, 5, 23, 25, 39, 44, 56, 63, 64, 66, 76, 76]

# Create instance of BoxAnnotator for drawing bounding boxes
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)     # Set speech rate
engine.setProperty('volume', 0.9)   # Set volume level

def speak_summary(counts):
    """
    Generate and speak out the summary of detected objects.
    
    Args:
        counts (Counter): A Counter object with class IDs and counts of detected objects.
    """
    # Generate a summary of detected objects
    summary = ", ".join(f"{CLASS_NAMES_DICT[class_id]} detected" for class_id in counts)
    
    # Use text-to-speech engine to say the summary
    engine.say(summary)
    engine.runAndWait()

def audio_thread(new_detections, stop_event):
    """
    Thread function to periodically speak out detected objects.
    
    Args:
        new_detections (Counter): A Counter object with new detections.
        stop_event (threading.Event): An event object to signal thread termination.
    """
    while not stop_event.is_set():
        # Speak out the detected objects if any are found
        if new_detections:
            speak_summary(new_detections)
            new_detections.clear()  # Clear the detections after speaking
        time.sleep(1)  # Wait for 1 second before checking again

# Open webcam for video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Adjust frame size for better performance
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Start the audio thread
new_detections = Counter()       # Counter for keeping track of new detections
stop_event = threading.Event()   # Event to signal stopping the audio thread
thread = threading.Thread(target=audio_thread, args=(new_detections, stop_event))
thread.daemon = True
thread.start()

# To keep track of previous counts for comparison
prev_counts = Counter()

# Main loop for processing video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

    # Model prediction on the resized frame
    try:
        results = model(small_frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Adjust detections to the original frame size
        detections.xyxy *= 2

        # Filter detections to only include selected classes
        detections = detections[np.isin(detections.class_id, selected_classes)]

        # Count objects by class for the current frame
        current_counts = Counter(detections.class_id)

        # Detect new objects or increases in counts
        for class_id, count in current_counts.items():
            if count > prev_counts.get(class_id, 0):
                new_detections[class_id] += count - prev_counts[class_id]

        # Update previous counts for the next iteration
        prev_counts = current_counts.copy()

        # Format custom labels with counts and confidence
        labels = [
            f"{CLASS_NAMES_DICT[class_id]} {current_counts[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate frame with bounding boxes and labels
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Display the annotated frame
        cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)

    except Exception as e:
        # Print any errors encountered during processing
        print(f"Error: {e}")

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Signal the audio thread to stop and wait for it to finish
stop_event.set()
