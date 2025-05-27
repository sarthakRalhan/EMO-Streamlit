import cv2
import numpy as np
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from batch_face import RetinaFace
from tensorflow.keras.preprocessing.image import img_to_array

class VideoCamera(object):
    def __init__(self, path_video='', path_report='', path_save='', name_labels = '', conf=0.7):
        self.path_video = path_video
        self.df = pd.read_csv(path_report)
        print(f"Loaded report with {len(self.df)} predictions")
        
        # Store emotion labels
        self.labels = name_labels
        print(f"Using emotion labels: {self.labels}")
        
        # Group predictions by frame
        self.frame_predictions = {}
        for _, row in self.df.iterrows():
            # Use simple frame numbers (1, 2, 3...)
            frame = str(int(row['frame']))
            if frame not in self.frame_predictions:
                self.frame_predictions[frame] = []
            
            # Get probabilities for all emotions
            probs = row[self.labels].values
            # Find the index of maximum probability
            max_idx = np.argmax(probs)
            
            self.frame_predictions[frame].append({
                'face_id': int(row['face_id']),
                'prob': probs,
                'best': [max_idx]  # Store only the index of maximum probability
            })
            # Sort by face_id to maintain order
            self.frame_predictions[frame].sort(key=lambda x: x['face_id'])
        
        print(f"Processed predictions for {len(self.frame_predictions)} frames")
        print(f"Sample frame predictions: {list(self.frame_predictions.items())[:2]}")
            
        self.path_save = path_save
        self.conf = conf
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.video = None
        self.vid_writer = None
        self.detector = RetinaFace(gpu_id=-1)

    def __del__(self):
        if self.video is not None:
            self.video.release()
        if self.vid_writer is not None:
            self.vid_writer.release()
        
    def draw_prob(self, emotion_yhat, best_n, startX, startY, endX, endY, face_id):
        # Get the top emotion and its probability
        top_emotion = self.labels[best_n[0]]
        top_prob = emotion_yhat[best_n[0]] * 100
        
        # Create label text
        label = f'Face {face_id}: {top_emotion}: {top_prob:.1f}%'
        
        # Calculate line width based on image size
        lw = max(round(sum(self.fr.shape) / 2 * 0.003), 2)
        
        # Pink color for both box and text
        pink_color = (255, 0, 255)  # BGR format for pink
        
        # Draw bounding box
        p1, p2 = (startX, startY), (endX, endY)
        cv2.rectangle(self.fr, p1, p2, pink_color, thickness=lw, lineType=cv2.LINE_AA)
        
        # Calculate text size and position
        tf = max(lw - 1, 1)  # Text thickness
        font_scale = 0.8  # Slightly smaller font for better fit
        text_size = cv2.getTextSize(label, self.font, font_scale, tf)[0]
        
        # Calculate text position
        text_x = p1[0] + (p2[0] - p1[0]) // 2 - text_size[0] // 2
        text_y = p1[1] - 10  # 10 pixels above the box
        
        # Adjust y position based on face_id to avoid overlapping text
        text_y -= face_id * 30  # 30 pixels offset per face
        
        # Draw text background for better visibility
        text_bg_color = (0, 0, 0)  # Black background
        cv2.rectangle(self.fr, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     text_bg_color, -1)
        
        # Draw text in pink
        cv2.putText(self.fr, label, (text_x, text_y), 
                    self.font, font_scale, pink_color, thickness=tf, lineType=cv2.LINE_AA)
            
    def get_video(self):
        print(f"\nProcessing video: {self.path_video}")
        self.video = cv2.VideoCapture(self.path_video)
        if not self.video.isOpened():
            print(f"Error: Could not open video file {self.path_video}")
            return
            
        total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = np.round(self.video.get(cv2.CAP_PROP_FPS))
        w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {total_frame} frames, {fps} FPS, {w}x{h}")
        
        self.path_save += '.mp4'
        self.vid_writer = cv2.VideoWriter(self.path_save, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        if not self.vid_writer.isOpened():
            print(f"Error: Could not create output video file {self.path_save}")
            return
            
        frame_count = 0
        while True:
            ret, self.fr = self.video.read()
            if not ret or self.fr is None:
                break
                
            frame_count += 1
            current_frame = str(frame_count)  # Use simple frame numbers
            
            # Get predictions for current frame
            frame_preds = self.frame_predictions.get(current_frame, [])
            if frame_count % 100 == 0:
                print(f"\nFrame {frame_count}:")
                print(f"Found {len(frame_preds)} predictions for this frame")
                print(f"Frame predictions: {frame_preds}")
            
            # Detect faces in current frame
            faces = self.detector(self.fr, cv=False)
            if frame_count % 100 == 0:
                print(f"Detected {len(faces)} faces in this frame")
                print(f"Face detection results: {faces}")
            
            # Process each detected face
            for f_id, box in enumerate(faces):
                box, _, prob = box
                if prob > self.conf and f_id < len(frame_preds):
                    startX = int(box[0])
                    startY = int(box[1])
                    endX = int(box[2])
                    endY = int(box[3])
                    
                    # Ensure coordinates are within frame bounds
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    
                    # Get prediction for this face
                    face_pred = frame_preds[f_id]
                    if frame_count % 100 == 0:
                        print(f"Drawing box for face {f_id} with prediction: {face_pred['best'][0]}")
                        print(f"Box coordinates: ({startX}, {startY}) to ({endX}, {endY})")
                        print(f"Emotion probabilities: {face_pred['prob']}")
                    self.draw_prob(face_pred['prob'], face_pred['best'], 
                                 startX, startY, endX, endY, face_pred['face_id'])
            
            # Write frame to output video
            self.vid_writer.write(self.fr)
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frame} frames")
                
        print(f"Finished processing {frame_count} frames")
        print(f"Output saved to: {self.path_save}")