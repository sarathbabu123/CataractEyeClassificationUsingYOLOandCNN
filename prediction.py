# Import necessary libraries
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import cv2 as cv
import numpy as np

# Load the pre-trained model
model = load_model("./Eyeclassification.h5")

# Define the labels for the classes
labels = {
    0: "Cataract",
    1: "Normal",
}

# Function to preprocess the image
def imagePreprocess(image):
    img = cv.imread(image)  # Read the image
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    img = cv.resize(img, (256, 256))  # Resize the image to 256x256
    img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)  # Normalize the image
    img = img.reshape(1, *img.shape)  # Reshape the image to match the input shape the model expects
    print(img.shape)
    return img

# Function to predict the class of the image
def prediction(image):
    image = imagePreprocess(image)  # Preprocess the image
    label = model.predict(image)  # Predict the class of the image
    label = np.argmax(label, axis=-1)  # Get the class with the highest probability
    label = labels[label[0]]  # Get the label of the class
    return label

# Function to run the YOLO model
def yolo():
    model = YOLO("./best (2).pt")  # Load the YOLO model
    
    cap = cv.VideoCapture(0)  # Start the video capture
    
    classNames = ["Cataract","Normal"]  # Define the class names
    while True:
        ret, frame = cap.read()  # Read a frame from the video
        results = model(frame,stream=True)  # Get the prediction results for the frame
        print(results)
        for result in results:
            boxes = result.boxes  # Get the bounding boxes

            for box in boxes:
                xyxy = box.xyxy[0].numpy()  # Get the coordinates of the bounding box
                x,y,x1,y1 = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
                print(x,y,x1,y1)
                cls = int(box.cls.numpy())  # Get the class of the object
                conf = box.conf[0].numpy()  # Get the confidence of the prediction
                conf = round((conf*100),2)
                label = f"{classNames[cls]},{conf}"  # Create the label
                cv.rectangle(frame,(x,y),(x1,y1),(0,255,0),3)  # Draw the bounding box on the frame
                cv.putText(frame,label,(x,y),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)  # Put the label on the frame

        ret,frame = cv.imencode(".jpg",frame)  # Encode the frame as a .jpg image
        frame = frame.tobytes()  # Convert the frame to bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Yield the frame as a multipart/x-mixed-replace response