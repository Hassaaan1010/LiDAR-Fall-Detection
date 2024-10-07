from ultralytics import YOLO
import cv2

pose_model = YOLO("../models/yolov8m-pose.pt")


def return_pose_map(pose_map):
    
    results = pose_model.predict(pose_map)
    
    num_people = 0
    
    for result in results:
        keypoints = result.keypoints
        if len(keypoints) > 0:  # Check if keypoints are detected
            num_people += 1  # Increment the count for each detected person
        
    keypoints = results[0].keypoints

    if len(keypoints) > 0:
        keypoints = keypoints[0].cpu().numpy()
    else:
        keypoints = None
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if keypoints is not None:
        xy = keypoints.data[0]
        for i in range(len(xy)):
            x, y, c = xy[i]
            if c > 0:
                cv2.circle(pose_map, (int(x), int(y)), 2, (0, 255, 0), -1)
                cv2.putText(
                    pose_map,
                    f"{i}-{int(c*100)}",
                    (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
    cv2.cvtColor(pose_map, cv2.COLOR_RGB2GRAY)
                
    return num_people, pose_map