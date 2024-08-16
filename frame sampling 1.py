# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:27:57 2023

@author: nagashree k d
"""
pip install opencv-python
import cv2

# Function to extract frames from a video
def extract_frames(video_path, output_folder, frames_per_sec=5):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Get the frames per second (fps) of the video
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))


    
    # Calculate the frame interval based on the desired frames per second
    frame_interval = int(fps / frames_per_sec)
    
    # Initialize frame counter
    frame_count = 0
    
    # Loop to read and save frames
    while True:
        ret, frame = video_capture.read()
        
        # Break the loop if we have reached the end of the video
        if not ret:
            break
        
        # Save the frame if it's time to capture according to the frame interval
        if frame_count % frame_interval == 0:
            frame_filename = f"{output_folder}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    # Release the video capture object
    video_capture.release()

if __name__ == "__main__":
    video_path = r"C:\Users\nagashree k d\Desktop\bird detection\video\video_20230804_141822\video_20230804_141822.mp4"# Path to your .mov video file
    output_folder = "frames 5"      # Output folder where frames will be saved
    frames_per_sec = 5     # Frames per second (adjust as needed)

    # Create the output folder if it doesn't exist
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Call the function to extract frames
    extract_frames(video_path, output_folder, frames_per_sec)


