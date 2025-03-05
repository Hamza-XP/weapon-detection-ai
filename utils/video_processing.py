import cv2
import os

def extract_key_frames(video_path, output_dir, interval=20):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            frame_name = f"frame_{frame_count}.jpg"
            output_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(output_path, frame)
            saved_frames.append(output_path)
        
        frame_count += 1
    
    cap.release()
    return saved_frames
