from ultralytics import YOLO
import cv2
import os
import numpy as np
print(os.getcwd())
try :
    from utils.check_overlap import check_overlap_for_2
except:
    from utils.check_overlap import check_overlap_for_2

pose_model = YOLO("./models/best.pt")
# pose_model = YOLO("./models/yolov8m-pose.pt")
# model = YOLO("./models/yolov8m.pt")



def check_fall(keypoints_data, bbox):
    
    bbox_verdict = False
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    threshold = width * 0.3  # threshold needs to be reletive to person bbox size
    
    # Check bbox
    if width > height * 1.2:  # increase to minimize false negatives (may also increase false positives.)
        bbox_verdict = True

    hip_left = keypoints_data[11]  
    hip_right = keypoints_data[12]  
    shoulder_left = keypoints_data[5]  
    shoulder_right = keypoints_data[6]
    # print("points : ",hip_left,hip_right, shoulder_left, shoulder_right)


    hip_left_x, hip_left_y, _ = hip_left
    hip_right_x, hip_right_y, _ = hip_right
    shoulder_left_x, shoulder_left_y, _ = shoulder_left
    shoulder_right_x, shoulder_right_y, _ = shoulder_right

    # average_hip_y = (hip_left_y + hip_right_y) / 
    # max hip height
    max_hip_y = min(hip_left_y, hip_right_y) if (hip_left_y != 0 and hip_right_y != 0) else max (hip_left_y, hip_right_y)
    if max_hip_y < 5 :
        return False
    
            
    # average_shoulder_y = (shoulder_left_y + shoulder_right_y) / 2
    # max shoulder height
    max_shoulder_y = max(shoulder_left_y, shoulder_right_y)
    if max_shoulder_y < 5:
        return False

    #height difference
    print("result",max_hip_y , max_shoulder_y + threshold, max_hip_y < max_shoulder_y+threshold )
    if max_hip_y < max_shoulder_y+threshold or bbox_verdict :  # Hips height higher than shoulders + threshold (threshold to minimize false negatives. Increase for lesser false negatives)
        return True



def return_pose_map(pose_map):  # pose_map is a frame
    copy_frame = np.copy(pose_map)  # Store the original frame
    variable = []  # To hold detected persons

    while True:
       
        # apply pose detection frame
        results = pose_model.predict(copy_frame, conf=0.50)  # Set confidence threshold as needed

        # process first person
        person = results[0]
        if len(person.boxes) == 0 or len(variable) == 5:  # no persons detected or 5 persons matched already
            break 
        
        # append detected person to the variable
        variable.append(person)  
        
        # bounding box of person
        keypoints = person.keypoints
        bbox = person.boxes.xyxy.cpu().numpy()[0]  #gets array of bboxes (only 1)
        x1, y1, x2, y2 = bbox
        
        
        # black out to avoid future detection
        cv2.rectangle(copy_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)


    num_people = 0
    num_fallen = 0

    # process each person data instance stored in "variable"
    for i,person in enumerate(variable):
        keypoints = person.keypoints
        
        if len(keypoints) > 0:  # Check if keypoints are detected
            num_people += 1  # Increment the count for each detected person
            keypoints_data = keypoints.data.cpu().numpy()[0]  # Get the first person's keypoints

            # Assuming the bounding box is available in the results for fallen check
            bbox = person.boxes.xyxy.cpu().numpy()  # Get bounding boxes

            if bbox.shape[0] > 0:
                # Check if the person is fallen
                if check_fall(keypoints_data, bbox[0]):
                    num_fallen += 1
                        
                    
                # Annotate the pose skeleton
                for keypoint in keypoints_data:  # Iterate through keypoints of the current person
                    x, y, c = keypoint  # Unpack coordinates and confidence
                    if c > 0:  # Only consider keypoints with confidence
                        cv2.circle(pose_map, (int(x), int(y)), 2, (0, 255, 0), -1)
                        cv2.putText(
                            pose_map,
                            f"{int(c * 100)}",
                            (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                        )

    print("Number of people detected:", num_people)
    print("Number of fallen people:", num_fallen)
    
    return num_people, num_fallen, pose_map



# def return_pose_map(pose_map):
    
#     results = pose_model.predict(pose_map)
    
#     num_people = 0
    
#     for person in results:
#         keypoints = person.keypoints
#         if len(keypoints) > 0:  # Check if keypoints are detected
#             num_people += 1  # Increment the count for each detected person
        
#     keypoints = results[0].keypoints

#     if len(keypoints) > 0:
#         keypoints = keypoints[0].cpu().numpy()
#     else:
#         keypoints = None
#     # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     if keypoints is not None:
#         xy = keypoints.data[0]
#         for i in range(len(xy)):
#             x, y, c = xy[i]
#             if c > 0:
#                 cv2.circle(pose_map, (int(x), int(y)), 2, (0, 255, 0), -1)
#                 cv2.putText(
#                     pose_map,
#                     f"{i}-{int(c*100)}",
#                     (int(x), int(y) - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (255, 0, 0),
#                     1,
#                 )
#     cv2.cvtColor(pose_map, cv2.COLOR_RGB2GRAY)
                
#     print("number of people detected: ",num_people)
#     print("len of results :",len(results))
#     # print(results)
#     return num_people, pose_map



# def return_pose_map(pose_map):  # pose_map is a frame
#     results = pose_model.predict(pose_map, conf=0.25)  # Set confidence threshold as needed

#     num_people = 0
#     num_fallen = 0

#     # Check if results are returned
#     if results:  
#         for person in results:
#             keypoints = person.keypoints
            
#             if len(keypoints) > 0:  # Check if keypoints are detected
#                 num_people += 1  # Increment the count for each detected person
#                 keypoints = keypoints.cpu().numpy()  # Convert to numpy array
                
#                 # Assuming the bounding box is available in the results for fallen check
#                 bbox = person.boxes.xyxy.cpu().numpy()  # Get bounding boxes

#                 if bbox.shape[0] > 0:
#                     for i in range(len(bbox)):
#                         x1, y1, x2, y2 = bbox[i]
#                         width = x2 - x1
#                         height = y2 - y1
                        
#                         # Check if the person is fallen
#                         if width > height * 1.5:  # Adjust the ratio as necessary
#                             num_fallen += 1
                        
#                         # Annotate the pose skeleton
#                         xy = keypoints[i]
#                         for j in range(len(xy)):
#                             x, y, c = xy[j]
#                             if c > 0:  # Only consider keypoints with confidence
#                                 cv2.circle(pose_map, (int(x), int(y)), 2, (0, 255, 0), -1)
#                                 cv2.putText(
#                                     pose_map,
#                                     f"{j}-{int(c * 100)}",
#                                     (int(x), int(y) - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX,
#                                     0.5,
#                                     (255, 0, 0),
#                                     1,
#                                 )
#     else:
#         print("No people detected in the frame.")

#     print("Number of people detected:", num_people)
#     print("Number of fallen people:", num_fallen)
#     cv2.cvtColor(pose_map, cv2.COLOR_RGB2GRAY)  # Convert the image to grayscale (if needed)
    
#     return num_people, num_fallen, pose_map

# Example usage:
# frame = ...  # Obtain your frame from a video source or image
# people_count, fallen_count, annotated_frame = return_pose_map(frame)


