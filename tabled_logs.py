import numpy as np
import cv2
import streamlit as st
import os
from datetime import datetime as dt
import pandas as pd

# Global list to store log messages
log_messages = []

def log(message, people_count):
    """Function to log messages with a timestamp and people count."""
    timestamp = dt.now().strftime("%H:%M:%S")  # Format time as HH:MM:SS
    log_messages.append((timestamp, message, people_count))  # Append a tuple of (time, message, people count)
    
    # Keep only the most recent 10 logs
    if len(log_messages) > 10:
        log_messages.pop(0)

def display_log(column):
    """Function to display log messages as a table in the 4th box."""
    # Create a DataFrame from the log messages
    

def main():
    st.title("YOLO Model with Camera Feed")


    # Set up the layout: 2 columns and 2 rows
    col1, col2 = st.columns([1, 1])  # Adjust column ratios to spread horizontally
    col3, col4 = st.columns([1, 1])  # Adjust column ratios for bottom row
    
    # Streamlit placeholders for the video feed
    st_frame1 = col1.empty()  # Top left
    st_frame2 = col2.empty()  # Top right
    st_frame3 = col3.empty()  # Bottom left
    
    # Add a checkbox to stop the video feed
    stop_feed = st.checkbox("Stop Camera Feed")

    while True:
        
        frame = cv2.imread("./2024-10-06-145518.jpg")  # Change to correct file path
        if frame is None:  # Check if frame was read successfully
            st.write("Error: Frame could not be read.")
            break

        if stop_feed:  # Stop the camera feed if the checkbox is checked
            st.write("Camera feed stopped")
            break
        
        # Resize the frame to be slightly bigger
        frame = cv2.resize(frame, (int(640 * 1.1), int(480 * 1.1)))  # Adjust size to your needs

        # Display the footage in three frames
        st_frame1.image(frame, channels="BGR", caption="Camera Output 1")
        st_frame2.image(frame, channels="BGR", caption="Camera Output 2")
        st_frame3.image(frame, channels="BGR", caption="Camera Output 3")
        
        
        randVar = np.random.randint(1, 15)
        fall = randVar == 10
        people_count = 4

        if fall:
            col4 = None
            
            timestamp = dt.now().strftime("%H:%M:%S")  # Format time as HH:MM:SS
            log_messages.append((timestamp, "Fall detected", people_count))  # Append a tuple of (time, message, people count)
            # Keep only the most recent 10 logs
            if len(log_messages) > 10:
                log_messages.pop(0)
                
            df = pd.DataFrame(log_messages, columns=["Time", "Message", "People"])
            # Adjust formatting for columns
            df["Time"] = df["Time"].str.pad(width=8, side='right', fillchar=' ')
            df["People"] = df["People"].astype(str).str.pad(width=2, side='left', fillchar=' ')
            
            _, col4 = st.columns([1,1])
            col4.empty()  # Clear the previous log
            col4.dataframe(df, use_container_width=True)  # Display as a DataFrame
                
            
    os._exit(0)  # Use to terminate the script
    

if __name__ == "__main__":
    main()
