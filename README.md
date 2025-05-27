# EMO-Streamlit

A Streamlit application for real-time emotion analysis in videos using the EMO AffectNet model.

## Features

- Upload and process video files
- Real-time emotion detection and analysis
- Multiple face detection and tracking
- Emotion statistics and visualization
- Interactive data display

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sarthakRalhan/EMO-Streamlit.git
cd EMO-Streamlit
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the required model files:
   - Create the following directory structure:
     ```
     models/
     ├── EmoAffectnet/
     │   ├── weights_0_66_37_wo_gl.h5
     │   └── torchscript_model_0_66_37_wo_gl.pth
     └── LSTM/
         └── RAVDESS_with_config.h5
     ```
   - Download the model files from [One Drive](https://adobe-my.sharepoint.com/:f:/p/sralhan/EpsJojkWRlpHuKNNAk8qqfsB-OfADYelzYciFRgszuE77A?e=dzmTSw) and place them in their respective directories

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload a video file through the web interface

4. View the emotion analysis results, including:
   - Processed video with emotion annotations
   - Emotion statistics for each detected face
   - Raw emotion data

## Project Structure

- `app.py`: Main Streamlit application
- `run.py`: Core emotion analysis script
- `visualization.py`: Video visualization script
- `models/`: Directory containing model weights (not included in repo)
- `functions/`: Helper functions and utilities

## Requirements

- Python 3.10
- See requirements.txt for full list of dependencies



