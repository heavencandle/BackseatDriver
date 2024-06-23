# BackseatDriver
> Interactive Autonomous Vehicle Control and Simulation

This project provides an interactive web application to control and simulate an autonomous vehicle in a virtual environment using natural language commands. The application uses Streamlit for the web interface, CARLA simulator for the virtual environment, and various AI models for natural language processing and computer vision.

## Features

- **Natural Language Commands:** Control the vehicle using natural language commands.
- **Object Detection:** Detect objects in the virtual environment using YOLOv5.
- **Scene Understanding:** Generate scene descriptions using CLIP.
- **Autonomous Vehicle Simulation:** Simulate vehicle behavior in the CARLA simulator.
- **Interactive Visualization:** Visualize the simulation and actions in real-time.

## Requirements

- Python 3.8 or later
- Streamlit
- OpenCV
- NumPy
- Pillow
- PyTorch
- Transformers
- CARLA

## Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/heavencandle/BackseatDriver
   cd your-project
   ```

2. Create and activate a virtual environment:

    ```
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    source .venv/bin/activate  # On macOS/Linux
    ```

3. Install the required packages:

    ```
    pip install streamlit opencv-python-headless numpy pillow torch transformers carla
    ```

4. Set up CARLA:
    - Download and extract the CARLA simulator from the CARLA releases page.
    - Follow the official CARLA documentation to set up and run the simulator.


## Usage
1. Start the CARLA simulator:
    Navigate to the directory where CarlaUE4.exe is located and run the executable.
    ```
    cd path\to\carla
    .\CarlaUE4.exe
    ```
2. Run the Streamlit application:
    ```
    streamlit run app.py
    ```
3. Interact with the application:
    - Use the web interface to upload images and enter natural language commands.
    - The application will display the simulation and the vehicle's actions based on the commands provided.

## Acknowledgments
CARLA Simulator
Streamlit
PyTorch
Hugging Face Transformers