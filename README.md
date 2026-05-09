# Waste-Classifier-System
The Waste Classifier System is a real-time computer vision application built entirely in Python using OpenCV, NumPy, and Tkinter. The project is designed to automatically detect and classify waste into different categories such as Organic Waste, Plastic Waste, and Person(to distinguish between waste & human) using a custom-built K-Nearest Neighbors (KNN) classification algorithm.
The system supports both **live webcam detection** and **image upload** classification, allowing users to identify waste types in real time. It uses handcrafted feature extraction techniques including color analysis, texture detection, and edge detection instead of relying on pre-trained deep learning models. The project was developed completely from scratch without using TensorFlow, PyTorch, or any external AI APIs.

FEATURES
Real-time webcam waste detection
Upload image classification mode
Classification into Organic, Plastic, and Person classes
Custom KNN algorithm implemented using NumPy
Confidence score calculation
Dynamic bounding box detection using MOG2 background subtraction
Smart caching system for faster startup
Modern dark-themed GUI built with Tkinter
Disposal tips based on detected waste type
Recent scan history tracking

Steps to run the Code:
1. Create a folder for the project to be downloaded.
2. Extract the files in that folder.
3. Inside the folder create another folder named "Dataset"
4. Inside "Dataset" Folder create another 3 folders: (this is to add new data for trainning)
        * Organic
        * Person
        * Plastic
5. Now you have to download some libraries in python (better use pycharm). The libraries are: 
        * OpenCV
        * NumPy
        * Tkinter
        * Pickle
6. Run on Pycharm or Any Other IDEs.
