# Saves the first 5 seconds of a video as frames [Best]
import cv2
import os

# Path to the video file
# video_path = '/Users/macos/Downloads/Football Stock Footages _ 1080p _ No Copyright 4K Zone (online-video-cutter.com).mp4'
# video_path = '/Users/macos/Downloads/Cars in Highway Traffic.mp4'
video_path = '/Users/macos/Desktop/InferenceEngine/video/output_video.mp4'
# Create a directory to save the frames
output_dir = '/Users/macos/Desktop/VideoFrame'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
time_limit = 5  # Time limit in seconds (fixed 5 seconds)
total_frames = int(fps * time_limit)  # Calculate the number of frames to process

print("Frames per second: ", fps)
print(f"Total frames to extract in the first {time_limit} seconds: {total_frames}")

frame_number = 0

while cap.isOpened() and frame_number < total_frames:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no more frames

    # Construct the file name
    frame_filename = f"{output_dir}/frame_{frame_number:04d}.png"
    
    # Save the frame as a PNG image
    cv2.imwrite(frame_filename, frame)
    
    frame_number += 1

# Release the video capture object
cap.release()
print("Frames extracted and saved.")
