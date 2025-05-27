import argparse
import numpy as np
import pandas as pd
import os
import time
from scipy import stats
from functions import sequences
from functions import get_face_areas
from functions.get_models import load_weights_EE, load_weights_LSTM

import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

parser = argparse.ArgumentParser(description="run")

parser.add_argument('--path_video', type=str, default='video/', help='Path to all videos')
parser.add_argument('--path_save', type=str, default='report/', help='Path to save the report')
parser.add_argument('--conf_d', type=float, default=0.7, help='Elimination threshold for false face areas')
parser.add_argument('--path_FE_model', type=str, default='models/EmoAffectnet/weights_0_66_37_wo_gl.h5',
                    help='Path to a model for feature extraction')
parser.add_argument('--path_LSTM_model', type=str, default='models/LSTM/RAVDESS_with_config.h5',
                    help='Path to a model for emotion prediction')

args = parser.parse_args()

def pred_one_video(path):
    start_time = time.time()
    label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    detect = get_face_areas.VideoCamera(path_video=path, conf=args.conf_d)
    dict_face_areas, total_frame = detect.get_frame()
    
    # Group faces by face_id
    faces_by_id = {}
    for face_key, face_data in dict_face_areas.items():
        face_id = face_data['face_id']
        if face_id not in faces_by_id:
            faces_by_id[face_id] = []
        faces_by_id[face_id].append(face_data)
    
    # Process each face separately
    all_predictions = []
    EE_model = load_weights_EE(args.path_FE_model)
    LSTM_model = load_weights_LSTM(args.path_LSTM_model)
    
    for face_id, face_data_list in faces_by_id.items():
        # Sort face data by frame number
        face_data_list.sort(key=lambda x: int(x['frame']))
        
        # Extract features for all frames of this face
        features_list = []
        for face_data in face_data_list:
            features = EE_model(np.expand_dims(face_data['face_area'], axis=0))
            features_list.append(features[0])  # Remove batch dimension
        
        # Create sequences of 10 frames with overlap
        sequence_length = 10
        for i in range(len(features_list)):
            # Get the last 10 frames (or pad with the first frame if not enough frames)
            start_idx = max(0, i - sequence_length + 1)
            sequence = features_list[start_idx:i + 1]
            
            # Pad the sequence if it's shorter than sequence_length
            if len(sequence) < sequence_length:
                sequence = [sequence[0]] * (sequence_length - len(sequence)) + sequence
            
            # Stack the sequence and add batch dimension
            sequence = np.stack(sequence)
            sequence = np.expand_dims(sequence, axis=0)
            
            # Get predictions for this sequence
            pred = LSTM_model(sequence).numpy()
            
            # Store prediction for the current frame
            pred_dict = {
                'frame': face_data_list[i]['frame'],
                'face_id': face_id,
                **{label: pred[0][i] for i, label in enumerate(label_model)}
            }
            all_predictions.append(pred_dict)
    
    # Create DataFrame with all predictions
    df = pd.DataFrame(all_predictions)
    
    # Sort by frame number and face_id
    df = df.sort_values(['frame', 'face_id'])
    
    # Save to CSV
    if not os.path.exists(args.path_save):
        os.makedirs(args.path_save)
        
    filename = os.path.basename(path)[:-4] + '.csv'
    df.to_csv(os.path.join(args.path_save, filename), index=False)
    
    end_time = time.time() - start_time
    print('Report saved in: ', os.path.join(args.path_save, filename))
    print('Lead time: {} s'.format(np.round(end_time, 2)))
    print()

def pred_all_video():
    path_all_videos = os.listdir(args.path_video)
    for id, cr_path in enumerate(path_all_videos):
        print('{}/{}'.format(id+1, len(path_all_videos)))
        pred_one_video(os.path.join(args.path_video, cr_path))
        
if __name__ == "__main__":
    pred_all_video()