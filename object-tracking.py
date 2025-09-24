import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time
import argparse
import os

# Set up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="yolo11l.pt", help="YOLOv11 model")  # Path to the YOLO model
parser.add_argument("--video", type=str, default="inference/videos/hotel.mp4", help="Path to input video or webcam index (0)")  # Input video file or webcam
parser.add_argument("--conf", type=float, default=0.25, help="Confidence Threshold for detection")  # Confidence threshold for object detection
parser.add_argument("--save", action="store_true", help="Save the result")  # Option to save the output video
args = parser.parse_args()  # Parse command-line arguments

# Function to display FPS (Frames Per Second) on the frame
def show_fps(frame, fps):
    x, y, w, h = 10, 10, 350, 50  # Define the position and size of the FPS display area
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)  # Draw a black rectangle for background
    cv2.putText(frame, "FPS: " + str(fps), (20, 52), cv2.FONT_HERSHEY_PLAIN, 3.5, (0, 255, 0), 3)  # Add FPS text to the frame

if __name__ == '__main__':
    # Set up video capture
    video_input = args.video  # Get the video input path from arguments
    if video_input.isdigit():  # Check if the input is a digit (indicating webcam)
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)  # Open webcam
    else:
        cap = cv2.VideoCapture(video_input)  # Open video file    

    # Save Video
    output_folder = "result"  # Directory to save the output video
    if(not os.path.isdir(output_folder)):  # Create the directory if it doesn't exist
        os.mkdir(output_folder)

    if args.save:  # If the save option is selected
        # Extract the filename from the input video and remove the extension
        filename = os.path.splitext(os.path.basename(args.video))[0]

        # Define the path for the output video
        output_video_path = f"{output_folder}/{filename}.mp4"  

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get frame width
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get frame height
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the input video

        # Create video writer objects to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))  # Initialize the VideoWriter

    conf_thres = args.conf  # Set confidence threshold for detection

    model = YOLO(args.model)  # Load the YOLOv11 model

    track_history = defaultdict(lambda: [])  # Initialize a dictionary to store track history of detected objects
    start_time = 0  # Initialize start time for FPS calculation

    class_id = 0  # Class ID for detection (set to 0 to track person)

    while cap.isOpened():  # Main loop to process video frames
        success, frame = cap.read()  # Read a frame from the video
        annotated_frame = frame  # Copy the frame for annotation

        if success:  # If the frame is read successfully
            # Perform object tracking using YOLO
            results = model.track(frame, classes=class_id, persist=True, tracker="bytetrack.yaml", conf=conf_thres, verbose=False)

            boxes = results[0].boxes.xywh.cpu()  # Get bounding boxes for detected objects

            if results[0].boxes.id is not None:  # Check if tracking IDs are available
                track_ids = results[0].boxes.id.int().cpu().tolist()  # Get tracking IDs
                class_ids = results[0].boxes.cls.int().cpu().tolist()  # Get class IDs

                # Plot the results on the frame
                annotated_frame = results[0].plot(line_width=2)

                # Draw tracking lines for each detected object
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x, y, w, h = box  # Extract box coordinates
                    track = track_history[track_id]  # Get the tracking history for the object
                    track.append((float(x), float(y)))  # Append the center point of the bounding box
                    if len(track) > 30:  # Retain track history for the last 30 frames
                        track.pop(0)  # Remove the oldest point if exceeding 30 points

                    # Draw the tracking lines on the annotated frame
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(255, 0, 0),  # Set color for tracking lines
                        thickness=3,
                    )

            # Calculate FPS
            end_time = time.time()  # Get the current time
            fps = 1 / (end_time - start_time)  # Calculate frames per second
            
            start_time = end_time  # Update start time for the next frame

            # Show FPS on the frame
            fps = float("{:.2f}".format(fps))  # Format FPS to two decimal places
            show_fps(annotated_frame, fps)  # Call function to display FPS

            resized_frame = cv2.resize(annotated_frame, (1280, 720))  # Resize frame for display        

            # Display the annotated frame in a fullscreen window
            cv2.namedWindow("YOLOv11 Person Tracking", cv2.WND_PROP_FULLSCREEN)  # Create a named window
            cv2.setWindowProperty("YOLOv11 Person Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set the window to fullscreen
            cv2.imshow("YOLOv11 Person Tracking", resized_frame)  # Show the annotated frame    

            if args.save:  # If the save option is selected
                writer.write(annotated_frame)  # Write the annotated frame to the output video file               

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  # Exit loop on 'q' key press
        else:
            # Break the loop if the end of the video is reached
            break

    if args.save:        
        print("The tracking results will be saved in: "+ output_video_path)  # Print the save location of the output video

    # Release the video capture object and close the display window
    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows