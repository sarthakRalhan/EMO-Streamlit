import streamlit as st
import tempfile
import os
import subprocess
import pandas as pd
import time
import shutil

# Set page config
st.set_page_config(
    page_title="EMO AffectNet Emotion Analysis",
    page_icon="😊",
    layout="wide"
)

# Title and description
st.title("EMO AffectNet Emotion Analysis")
st.markdown("""
This app analyzes emotions in videos using the EMO AffectNet model. Upload a video to see the emotion analysis results.
""")

def process_video(video_file):
    """Process the uploaded video using run.py and visualization.py"""
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    try:
        # Create necessary subdirectories
        video_dir = os.path.join(temp_dir, "video")
        report_dir = os.path.join(temp_dir, "report")
        result_dir = os.path.join(temp_dir, "result_videos")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        
        # Save uploaded video
        video_path = os.path.join(video_dir, "input_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Run emotion analysis using run.py
        with st.spinner('Analyzing emotions...'):
            run_cmd = [
                "python", "run.py",
                "--path_video", video_dir,
                "--path_save", report_dir,
                "--conf_d", "0.7",
                "--path_FE_model", "models/EmoAffectnet/weights_0_66_37_wo_gl.h5",
                "--path_LSTM_model", "models/LSTM/RAVDESS_with_config.h5"
            ]
            subprocess.run(run_cmd, check=True)
        
        # Generate visualization using visualization.py
        with st.spinner('Generating visualization...'):
            vis_cmd = [
                "python", "visualization.py",
                "--path_video", video_dir,
                "--path_report", report_dir,
                "--path_save_video", result_dir,
                "--conf_d", "0.7"
            ]
            subprocess.run(vis_cmd, check=True)
        
        # Get the output video path
        output_video = os.path.join(result_dir, "input_video.mp4")
        
        # Read the CSV report
        report_path = os.path.join(report_dir, "input_video.csv")
        df = pd.read_csv(report_path)
        
        return df, output_video, temp_dir
    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir)
        raise e

def main():
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Process video
        with st.spinner('Processing video... This may take a few minutes.'):
            df, output_video_path, temp_dir = process_video(uploaded_file)
        
        try:
            # Display results
            st.success('Video processing complete!')
            
            # Show video
            st.subheader("Processed Video with Emotion Analysis")
            with open(output_video_path, 'rb') as video_file:
                st.video(video_file)
            
            # Show emotion statistics
            st.subheader("Emotion Statistics")
            
            # Calculate average emotions per face
            emotion_columns = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
            avg_emotions = df.groupby('face_id')[emotion_columns].mean()
            
            # Display emotion charts
            for face_id in avg_emotions.index:
                st.write(f"Face {face_id} Emotion Distribution")
                st.bar_chart(avg_emotions.loc[face_id])
            
            # Show raw data
            st.subheader("Raw Emotion Data")
            st.dataframe(df)
        finally:
            # Clean up temporary directory after we're done with the video
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()

