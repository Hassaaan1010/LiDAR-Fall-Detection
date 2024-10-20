import numpy as np
import cv2
import streamlit as st
import pandas as pd
from datetime import datetime as dt
from utils.pose_mapping import return_pose_map
import os
from utils.monodepth import load_model, depth_map_from_frame

dpt_model, transform, device = load_model()


log_messages = []

def main():
    st.title("YOLO Model with Camera Feed")

    # Set up the layout: 2 columns and 2 rows
    col1, col2 = st.columns([2, 2])  # Adjust column ratios to spread horizontally
    col3, col4 = st.columns([2, 2])  # Adjust column ratios for bottom row

    # Streamlit placeholders for the video feed
    st_frame1 = col1.empty()  # Top left
    st_frame2 = col2.empty()  # Top right
    st_frame3 = col3.empty()  # Bottom left (for the third image)
    log_placeholder = col4.empty()  # Placeholder for logs (bottom right)

    # Add a checkbox to stop the video feed
    stop_feed = st.checkbox("Stop Camera Feed")
    
    
    while not stop_feed:
        
        frame = cv2.imread("./sample_images/473.png")  
        
        # depth_map = cv2.imread("./output.jpg")
        depth_map = np.copy(frame)
        depth_map = depth_map_from_frame(depth_map, dpt_model, transform, device)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        
        pose_map = np.copy(depth_map)
        
        num_of_people , num_fallen,  pose_map = return_pose_map(pose_map)
              
        if frame is None:  # Check if frame was read successfully
            st.write("Error: Frame could not be read.")
            break

        if stop_feed:  # Stop the camera feed if the checkbox is checked
            st.write("Camera feed stopped")
            break
                    
        
        # Display the footage in three frames
        st_frame1.image(frame, channels="BGR", caption="Camera Output 1")
        st_frame2.image(depth_map, channels="BGR", caption="Depthoutput Output 2")
        st_frame3.image(pose_map, channels="BGR", caption="Pose detection Output 3")  # Added back the third frame
        
        # Simulate fall detection
        fall = num_fallen    # 0 if no fall, greater than 0 if fallen...

        if fall:
            timestamp = dt.now().strftime("%H:%M:%S")  # Format time as HH:MM:SS
            log_messages.append((timestamp, "Fall detected", num_fallen))  # Append a tuple of (time, message, people count)

            # Keep only the most recent 7 logs
            if len(log_messages) > 7:
                log_messages.pop(0)
        
            # Create a DataFrame from the log messages
            df = pd.DataFrame(log_messages, columns=["Time", "Message", "Fallen"])
            df["Time"] = df["Time"].str.pad(width=8, side='right', fillchar=' ')
            df["Fallen"] = df["Fallen"].astype(str).str.pad(width=2, side='left', fillchar=' ')

            # Clear and update the log table
            log_placeholder.empty()  # Clear the previous logs
            log_placeholder.write(df)  # Display the updated DataFrame

        
    # cap.release()
    os._exit(0)

if __name__ == "__main__":
    main()


