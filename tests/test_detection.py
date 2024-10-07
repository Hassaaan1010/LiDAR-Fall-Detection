import cv2
import torch

from utils.monodepth import load_model, depth_map_from_frame

# Load the model
dpt_model, transform, device = load_model()


def apply_pose_detection(frame, model):
    """
    Applies pose detection using a trained YOLOv8 model on the provided frame.

    Args:
        frame (numpy.ndarray): The input frame from the camera.
        model (torch.nn.Module): The loaded YOLOv8 pose detection model.

    Returns:
        tuple: A tuple containing:
            - num_people (int): Number of detected people.
            - processed_frame (numpy.ndarray): The frame with bounding boxes and pose detections.
    """
    # Convert the frame to a format suitable for the model
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get the depth map
    depth_map = depth_map_from_frame(img_rgb, model, transform, device)

    
    # Perform inference
    results = model(img_rgb)
    
    # Extract bounding boxes and pose detections
    detections = results.pandas().xyxy[0]  # Get detections in a pandas DataFrame format
    num_people = len(detections)  # Number of detected people

    # Draw bounding boxes and pose landmarks on the frame
    for _, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls = row[:6]
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        # Optionally: Draw pose keypoints if needed (requires separate keypoint detection)
        # Here, you would call a function or process to draw keypoints for each detected person

    return num_people, frame

# Usage example:
# Load your trained YOLOv8 model (adjust the path as necessary)
model = torch.hub.load('ultralytics/yolov8', 'custom', 'path/to/your/yolov8m-pose.pt')

# Capture frame from your camera or video source
# cap = cv2.VideoCapture(0)  # Use 0 for the webcam or provide a video file path
# ret, frame = cap.read()
ret = True
frame = cv2.imread("/home/hassaan/Downloads/train_2500/images_final/17.png")


if ret:
    num_people, processed_frame = apply_pose_detection(frame, model)
    print(f"Number of detected people: {num_people}")

    # Display the processed frame
    cv2.imshow('Processed Frame', processed_frame)
    cv2.waitKey(0)

# cap.release()
cv2.destroyAllWindows()
