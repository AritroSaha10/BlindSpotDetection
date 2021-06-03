import numpy as np
import cv2
from PIL import Image
from time import time

import tflite_runtime.interpreter as tflite

# Image processing function
def processImg(img):
    img_tensor = np.array(img).astype(np.float32)                # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)              # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    # img_tensor /= 255.                                           # expects values in the range [0, 1]

    return img_tensor

# Sigmoid function for transforming model output
def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="mobilenetv2_BSD.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video cap
cams = [cv2.VideoCapture(0), cv2.VideoCapture(2)] # Check v4l2-ctl --list-devices for cam ids

for i, cam in enumerate(cams):
    if not cam.isOpened():
        print(f"Couldn't open camera {i}")
        exit()

font = cv2.FONT_HERSHEY_SIMPLEX

def getPredictionFromRetAndFrame(ret, frame, winName):
    if not ret:
        print("Can't recieve frame, exiting...")
        exit()
    
    timer = time()

    # Draw box first
    # overlay = frame.copy()
    # output = frame.copy()

    LINE_THICKNESS = 10
    LINE_COLOUR    = (0, 0, 255)
    ALPHA          = 0.5
    
    # Top layer
    cv2.line(frame,   (0,     678),     (1280, 450),    LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1280,  450),     (1750, 450),    LINE_COLOUR,    thickness=LINE_THICKNESS)

    # Farther side layer
    cv2.line(frame,   (1280,  450),     (1280, 656),    LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1750,  450),     (1739, 627),    LINE_COLOUR,    thickness=LINE_THICKNESS)

    # Bottom layer
    cv2.line(frame,   (1739,  627),     (1254, 1080),   LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1280,  656),     (0,    1080),   LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1280,  656),     (1739,  627),   LINE_COLOUR,    thickness=LINE_THICKNESS)
    
    # Apply overlay with 50% transparency to original frame, takes 0.5s
    # cv2.addWeighted(overlay, ALPHA, output, 1 - ALPHA, 0, output)
    
    # Crop frame and resize
    cropped = frame[360:1080, 0:1920]
    resized = cv2.resize(cropped, (160, 160))

    # Preprocess image and predict
    interpreter.set_tensor(input_details[0]['index'], processImg(resized))
    interpreter.invoke()

    # Convert raw to confidence level using sigmoid func
    pred_raw = interpreter.get_tensor(output_details[0]['index'])
    pred_sig = sigmoid(pred_raw)
    pred = np.where(pred_sig < 0.5, 0, 1)
    timer = time() - timer

    # Print results
    # readable_val = f"Car in box with {round((1 - pred_sig[0][0]) * 100, 2)}% confidence" if pred[0][0] == 0 else f"No car in box with {round(pred_sig[0][0] * 100, 2)}% confidence"
    # readable_val = f"Car in box" if pred[0][0] == 0 else f"No car in box"
    # print(f"Preprocessing + Prediction took {round(timer, 3)}s")
    # print(readable_val)
    readable_val = winName if pred[0][0] == 0 else ""
    print(readable_val)
    print("----------------------\n\n")

    print()

    # Put some text for the user to see and show the frame
    # small_frame = cv2.resize(frame, (1280, 720))
    # cv2.putText(small_frame, readable_val, ((1280//2) - 250, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # cv2.imshow(winName, small_frame)

cameraNames = ["Left", "Right"]

# Event loop
while True:
    for i in range(len(cams)):
        ret, frame = cams[i].read()
        getPredictionFromRetAndFrame(ret, frame, cameraNames[i])

    # Quit on q
    if cv2.waitKey(1) == ord('q'):
        break
