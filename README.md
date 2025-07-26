# Philippine Motorcycle License Plate Optical Character Recognition (OCR) using Huawei Cloud with Nvidia Jetson Nano and MongoDB 

## Overview
This project focuses on the application of Huawei Cloud’s Optical Character Recognition (OCR) in the development of a real-time motorcycle license plate classification system with Nvidia Jetson Nano and MongoDB. This project uses real-time images of Philippine motorcycle license plates to determine and classify the region of registration and status of the vehicle.

## Objectives

### General Objective:
To develop an accurate real-world motorcycle license plate classification system, that identifies the region and allows the user to set the current status of the vehicle, using Huawei Cloud’s OCR, optimized for deployment on Nvidia Jetson Nano and MongoDB. 

### Specific Objectives: 

- To train, test, and apply Huawei Cloud’s OCR model that classifies motorcycle license plates in real-time.
- To develop a functional application where information (i.e., which Philippine region the vehicle was registered at and if the vehicle is blacklisted, expired, etc.) regarding the classified motorcycle license plates can be set and displayed. 
- To deploy the text recognition system on an edge device, Jetson Nano, ensuring real-time performance and efficiency on edge devices. 
- To implement a MongoDB database system for efficient storage and retrieval of motorcycle license plate recognition data.

## Requirements

Before running the application, make sure the following are installed:

### 1. Download and install [Python 3.9.19](https://www.python.org/downloads/release/python-3919/)

### 2. Set up a Python Virtual Environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install packages via Terminal:
    pip install -r requirements.txt

### 4. Set up a Cloud Database in [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)

### 5. Modify the Database lines on _mongodb_test.py_ to connect your created Cloud Database