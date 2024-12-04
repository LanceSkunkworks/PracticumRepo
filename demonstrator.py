import cv2
import os
import numpy as np
import tensorflow as tf

# Paths and directories
LITE_RT_EXPORT_PATH = "vehicle-detection-angled-2024102800/2024102800_v10_landscape_tram_416/"
LITE_RT_MODEL = 'best_float32.tflite'
LITE_RT_MODEL_PATH = LITE_RT_EXPORT_PATH + LITE_RT_MODEL

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=LITE_RT_MODEL_PATH)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()

# Path to the video file
video_path = '/Users/macos/Downloads/Cars in Highway Traffic.mp4'
# video_path = '/Users/macos/Downloads/highwaycars.mp4'
output_video_path = 'video/OutputVideo.mp4'
# Directory to save individual frames (optional)
output_dir = '/Users/macos/Desktop/VideoFrame'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
time_limit = 5  # Time limit in seconds (fixed 5 seconds)
total_frames = int(fps * time_limit)  # Calculate the number of frames to process

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_number = 0
frames = []

print("Frames per second: ", fps)
print(f"Total frames to extract in the first {time_limit} seconds: {total_frames}")

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Preprocess the image
def preprocess(img, input_shape):
    target_width, target_height = input_shape[2], input_shape[1]
    img = cv2.resize(img, (target_width, target_height), cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0) * 2 - 1
    return np.expand_dims(img, axis=0)

def drawBox(data, img, img_shape):
    img_height, img_width, _ = img_shape
    rounded_score = data[5]
    if data[4] == 0:
        obj_class = 'Car' # Checks if the class is equal to 0(Car)
    elif data[4] != 0:
        obj_class = 'Unknown'
    x_min, x_max = int(data[1] * img_width), int(data[3] * img_width)
    y_min, y_max = int(data[0] * img_height), int(data[2] * img_height)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    cv2.putText(img, f'{obj_class} {"{:.2f}".format(rounded_score)}', (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
    return img

while cap.isOpened() and frame_number < total_frames:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no more frames

    # Preprocess the frame
    input_data = preprocess(frame, input_details[0]['shape'])

    # Run model inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve output data  
    boundingBox = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    classes = np.expand_dims(interpreter.get_tensor(interpreter.get_output_details()[1]['index']), axis=-1)
    scores = np.expand_dims(interpreter.get_tensor(interpreter.get_output_details()[2]['index']), axis=-1)
    output_data = np.squeeze(np.concatenate((boundingBox, classes, scores), axis=-1))

    # Draw bounding boxes on the frame
    for output in output_data:
        frame = drawBox(output, frame, frame.shape)

    # Collect the frame for video
    frames.append(frame)

    # Write the frame to the video file
    out_video.write(frame)

    # Optional visualization
    cv2.imshow('Frame with Bounding Boxes', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1  # Increment frame count

# Release resources
cap.release()
out_video.release()
cv2.destroyAllWindows()
