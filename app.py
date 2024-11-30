import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO

# Load the trained model
model = load_model('model.h5')  # Model file in the same directory as app.py

# Define the function to extract frames from the uploaded video
def extract_frames_from_video(video_file, desired_fps=30, img_size=(224, 224)):
    """Extract frames from the uploaded video file."""
    # Write the video to a temporary buffer in memory
    video_bytes = video_file.read()
    video_buffer = BytesIO(video_bytes)

    # Open the video with OpenCV
    cap = cv2.VideoCapture(video_buffer)
    frames = []
    
    # Get original FPS of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0 or not original_fps:
        st.error("Could not determine FPS for the video. Skipping...")
        cap.release()
        return []

    # Calculate frame interval to match desired FPS
    frame_interval = int(original_fps / desired_fps)
    if frame_interval == 0:
        frame_interval = 1  # Ensure at least one frame is saved per iteration
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            # Resize the frame to the target size
            resized_frame = cv2.resize(frame, img_size)
            frames.append(resized_frame)
        count += 1

    cap.release()
    return np.array(frames)

# Define the function to preprocess frames and predict using the model
def predict_video_class(frames):
    """Preprocess frames and predict using the trained model."""
    # Normalize frames to the range [0, 1]
    frames = frames.astype('float32') / 255.0

    # Make predictions using the trained model
    predictions = model.predict(frames)

    # Get the class with the highest probability for each frame
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Count the number of occurrences of each class (Yawning or Microsleep)
    class_counts = [np.sum(predicted_classes == i) for i in range(len(predicted_classes))]

    return predicted_classes, class_counts

# Streamlit app layout
st.title("Video Classification: Yawning or Microsleep")

st.sidebar.header("Upload Video")

# Upload the video
video_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if video_file:
    st.video(video_file)  # Display the uploaded video

    # Extract frames from the video
    st.write("Extracting frames from the video...")
    frames = extract_frames_from_video(video_file)
    
    if len(frames) > 0:
        st.write(f"Extracted {len(frames)} frames from the video.")
        
        # Predict the class for each frame
        st.write("Making predictions on the frames...")
        predicted_classes, class_counts = predict_video_class(frames)
        
        # Find the class with the maximum number of frames
        major_prediction = np.argmax(class_counts)
        major_class = ["Yawning", "Microsleep"][major_prediction]
        major_class_count = class_counts[major_prediction]

        # Display the final video prediction
        st.write(f"Predicted class for the video: {major_class}")
        st.write(f"Number of frames contributing to this prediction: {major_class_count}")
        
        # Show a pie chart for the distribution of predicted classes
        st.write("Class distribution (Yawning vs Microsleep):")
        st.write(f"Yawning: {class_counts[0]} frames")
        st.write(f"Microsleep: {class_counts[1]} frames")
        
        st.bar_chart(class_counts)
    else:
        st.error("No frames were extracted from the video. Please try a different video.")
