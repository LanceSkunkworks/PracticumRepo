#the original yolov8 object detection
#NO BLACK BACKGROUND
import cv2 as cv
from ultralytics import YOLO

def processVid():
    # Path to the video
    path = '/Users/macos/Downloads/5secSoccer.mp4'
    
    # Open the video capture
    vs = cv.VideoCapture(0)

    if not vs.isOpened():
        print("Error: Could not open video.")
        return
    
    # Load the YOLOv8 nano model
    model = YOLO("yolov8n.pt")  # Ensure this is correct and the model exists

    while True:
        # Read the next frame
        (grabbed, frame) = vs.read()

        # If no frame was grabbed (end of video), reset to loop
        if not grabbed:
            vs.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        # Use YOLO model for predictions
        results = model.predict(frame, stream=False, classes = [0]) #classes depends which objects you want to detech e.g. a ball or a person
        for result in results:
            for data in result.boxes.data.tolist():
                print(data)
                drawBox(data, frame)
                # You can implement your drawBox function here to draw boxes on the frame

        # Resize the frame to 960x540
        video_resized = cv.resize(frame, (960, 540))
        flipped = cv.flip(video_resized,1)
        # Display the resized frame
        cv.imshow("Object Detection", flipped)

        

        # Wait for 24ms before showing the next frame (for approximately 24 fps)
        if cv.waitKey(5) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    vs.release()
    cv.destroyAllWindows()

def drawBox(data, image): #data is from processVid() function
    
    x1, y1, x2, y2, conf, id = data #we detect individual models with id. x and y are coordinates respectively
    p1 = int(x1), int(y1)
    p2 = int(x2), int(y2)
    cv.rectangle(image, p1, p2, (0, 255,0), 3)

    return image

    

# Call the function to process the video
processVid()