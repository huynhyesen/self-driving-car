
# Self-Driving Car using Deep Learning

A self-driving car simulation project using
deep learning and the Udacity Self-Driving Car Simulator.

## Overview

This project implements a self-driving car system
using PilotNet to predict steering angles from
front-facing camera images.

The system communicates with the simulator
through Socket.IO.

## Architecture

Simulator
    |
    v
Camera Image
    |
    v
Preprocessing
    |
    v
PilotNet (CNN)
    |
    v
Steering Prediction
    |
    v
Control Command
    |
    v
Socket.IO
    |
    v
Simulator

## Technologies

- Python 3.10
- PyTorch
- OpenCV
- NumPy
- Socket.IO
- Udacity Self-Driving Car Simulator

## Project Structure

- drive.py: Simulator communication and inference
- model.py: PilotNet architecture
- utils.py: Utility functions
- models/: Trained model weights

## Installation

```bash
git clone git@github.com:huynhyesen/self-driving-car.git
cd self-driving-car

python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

1. Start the Udacity simulator.
2. Run the driving server:

```bash
python drive.py
```

## Model

PilotNet is a convolutional neural network
used for steering angle prediction.

## Acknowledgements

- Udacity Self-Driving Car Simulator
- NVIDIA PilotNet research
