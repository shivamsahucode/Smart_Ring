Smart Ring – IMU Gesture Recognition
Overview

This project implements an IMU-based Smart Ring for hand gesture recognition. The system collects motion data from an IMU sensor, processes accelerometer and gyroscope readings, trains a machine learning model, and performs real-time gesture prediction.

Features
IMU sensor data collection
Accelerometer and gyroscope motion analysis
Gesture dataset creation
Machine learning model training
Real-time gesture recognition
Wearable Smart Ring-based interaction
Project Structure
Smart_Ring/
│
├── IMU_Gesture_Data/      # Collected IMU gesture dataset
├── data_collector.py      # Collects gesture data from the IMU sensor
├── model_train.py         # Trains the gesture recognition model
├── new_py_v4.py           # Gesture recognition implementation
├── new_rt_v2.py           # Real-time gesture recognition
├── rt_v5.py               # Real-time prediction implementation
├── v3_rtmodel.py          # Real-time trained model implementation
└── README.md
Workflow
IMU Sensor
    ↓
Data Collection
    ↓
IMU Gesture Dataset
    ↓
Data Processing
    ↓
Model Training
    ↓
Trained Model
    ↓
Real-Time Gesture Recognition
Installation

Clone the repository:

git clone https://github.com/shivamsahucode/Smart_Ring.git
cd Smart_Ring

Install the required Python libraries:

pip install numpy pandas scikit-learn

Install any additional dependencies required by the specific sensor communication setup.

Usage
1. Collect Gesture Data

Run:

python data_collector.py

The collected IMU data is stored in the IMU_Gesture_Data directory.

2. Train the Model

Run:

python model_train.py

This processes the collected gesture data and trains the gesture recognition model.

3. Real-Time Gesture Recognition

Run one of the real-time implementations:

python rt_v5.py

or:

python new_rt_v2.py

The system reads live IMU sensor data and predicts the performed gesture.

Technologies Used
Python
IMU Sensor
Accelerometer
Gyroscope
NumPy
Pandas
Scikit-learn
Machine Learning
Applications
Gesture-controlled systems
Human-Computer Interaction
Wearable technology
Smart devices
Assistive technology
Future Improvements
Improve gesture recognition accuracy
Add support for more gestures
Optimize the model for faster real-time prediction
Reduce hardware size and power consumption
Integrate with mobile or IoT applications
