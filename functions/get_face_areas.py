import cv2
import numpy as np
import os
from functions import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.image import img_to_array
from batch_face import RetinaFace

class VideoCamera(object):
    def __init__(self, path_video='', conf=0.7):
        self.path_video = path_video
        self.conf = conf
        self.cur_frame = 0
        self.video = None
        self.dict_face_area = {}
        self.detector = RetinaFace(gpu_id=-1)

    def __del__(self):
        if self.video is not None:
            self.video.release()
        
    def preprocess_image(self, cur_fr):
        cur_fr = utils.preprocess_input(cur_fr, version=2)
        return cur_fr
        
    def channel_frame_normalization(self, cur_fr):
        cur_fr = cv2.cvtColor(cur_fr, cv2.COLOR_BGR2RGB)
        cur_fr = cv2.resize(cur_fr, (224,224), interpolation=cv2.INTER_AREA)
        cur_fr = img_to_array(cur_fr)
        cur_fr = self.preprocess_image(cur_fr)
        return cur_fr
            
    def get_frame(self):
        print()
        print(self.path_video)
        self.video = cv2.VideoCapture(self.path_video)
        if not self.video.isOpened():
            print(f"Error: Could not open video file {self.path_video}")
            return {}, 0
            
        total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = np.round(self.video.get(cv2.CAP_PROP_FPS))
        w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print('Name video: ', os.path.basename(self.path_video))
        print('Number total of frames: ', total_frame)
        print('FPS: ', fps)
        print('Video duration: {} s'.format(np.round(total_frame/fps, 2)))
        print('Frame width:', w)
        print('Frame height:', h)
        
        frame_count = 0
        while True:
            ret, self.fr = self.video.read()
            if not ret or self.fr is None: 
                break
                
            frame_count += 1
            name_img = str(frame_count).zfill(6)
            
            # Detect faces in current frame
            faces = self.detector(self.fr, cv=False)
            
            # Process all detected faces
            for f_id, box in enumerate(faces):
                box, _, prob = box
                if prob > self.conf:
                    startX = int(box[0])
                    startY = int(box[1])
                    endX = int(box[2])
                    endY = int(box[3])
                    
                    # Ensure coordinates are within frame bounds
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    
                    # Extract and process face region
                    cur_fr = self.fr[startY: endY, startX: endX]
                    if cur_fr.size == 0:  # Skip if face region is empty
                        continue
                        
                    # Store each face with a unique key combining frame number and face ID
                    face_key = f"{name_img}_face_{f_id}"
                    self.dict_face_area[face_key] = {
                        'frame': name_img,
                        'face_id': f_id,
                        'face_area': self.channel_frame_normalization(cur_fr),
                        'bbox': (startX, startY, endX, endY)
                    }
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frame} frames")
                
        print(f"Finished processing {frame_count} frames")
        del self.detector          
        return self.dict_face_area, total_frame