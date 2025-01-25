# FACE-FINDER

## Overview
FACE-FINDER is a face detection and recognition application that uses computer vision and machine learning to identify and locate faces in images or video streams. The project includes a preprocessing script to prepare data and a Streamlit-based web application for user interaction.

---

## Features
- **Face Detection**: Detects faces in images or video frames.
- **Face Recognition**: Recognizes known faces using a pre-trained model.
- **Preprocessing**: Prepares and cleans data for better model performance.
- **Streamlit App**: Provides an interactive web interface for uploading images/videos and viewing results.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Parama-Guru/FACE-FINDER.git
   cd FACE-FINDER
   cd another_approach
2. **Preprocessin phase**:
   change the path to your directory consist of photos
   ```bash
   python preprocess.py
3. **Run the Application**:
   change the path of the pickle file if needed
   ```bash
   streamlit run app.py
