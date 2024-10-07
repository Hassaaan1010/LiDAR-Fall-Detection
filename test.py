import numpy as np
import cv2
import streamlit as st
import pandas as pd
from datetime import datetime as dt

# Global list to store log messages
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

    while True:
        frame = cv2.imread("./2024-10-06-145518.jpg")  # Change to correct file path
        if frame is None:  # Check if frame was read successfully
            st.write("Error: Frame could not be read.")
            break

        if stop_feed:  # Stop the camera feed if the checkbox is checked
            st.write("Camera feed stopped")
            break
        
        # Resize the frame to be slightly bigger
        frame = cv2.resize(frame, (int(640 * 2), int(480 * 2)))  # Adjust size to your needs

        # Display the footage in three frames
        st_frame1.image(frame, channels="BGR", caption="Camera Output 1")
        st_frame2.image(frame, channels="BGR", caption="Camera Output 2")
        st_frame3.image(frame, channels="BGR", caption="Camera Output 3")  # Added back the third frame
        
        # Simulate fall detection
        randVar = np.random.randint(1, 30)
        fall = randVar == 10
        people_count = 4

        if fall:
            timestamp = dt.now().strftime("%H:%M:%S")  # Format time as HH:MM:SS
            log_messages.append((timestamp, "Fall detected", people_count))  # Append a tuple of (time, message, people count)

            # Keep only the most recent 10 logs
            if len(log_messages) > 10:
                log_messages.pop(0)
        
            # Create a DataFrame from the log messages
            df = pd.DataFrame(log_messages, columns=["Time", "Message", "People"])
            df["Time"] = df["Time"].str.pad(width=8, side='right', fillchar=' ')
            df["People"] = df["People"].astype(str).str.pad(width=2, side='left', fillchar=' ')
            
            # Clear and update the log table
            log_placeholder.empty()  # Clear the previous logs
            log_placeholder.write(df)  # Display the updated DataFrame

if __name__ == "__main__":
    main()


