!! TODO: check communication between app.py and webots, and make robots / vehicle to move

import os
import sys
os.environ['WEBOTS_HOME'] = 'C:\\Program Files\\Webots'
os.environ['PYTHONPATH'] = os.path.join(os.environ['WEBOTS_HOME'], 'lib\\controller\\python') + ';' + os.environ.get('PYTHONPATH', '')
sys.path.append(os.path.join(os.environ['WEBOTS_HOME'], 'lib\\controller\\python'))
import streamlit as st
import cv2
import numpy as np
import subprocess
import time
import os
from threading import Thread
from transformers import pipeline
from controller import Robot, Camera

# Configuration
TIME_STEP = 32
OUTPUT_DIR = "/tmp/webots_data"
COMMAND_FILE = "/tmp/command.txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize NLP model
nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define possible commands
commands = ["move forward", "turn left", "turn right", "stop"]

# Streamlit setup
st.set_page_config(page_title="Autonomous Driving Simulator", page_icon="🚗")
st.title("Autonomous Driving Simulator with NLP Control")
st.markdown("""
This application allows you to control a Webots simulation using natural language commands. Enter a command like "move forward", "turn left", "turn right", or "stop" to control the robot.
""")

# Text input for commands
user_command = st.text_input("Enter command:", "")

if user_command:
    # Classify the command
    result = nlp_model(user_command, commands)
    command = result['labels'][0]

    # Write command to file
    with open(COMMAND_FILE, "w") as f:
        f.write(command)
    st.write(f"Command '{command}' sent to the robot.")

# Function to get the latest frame
def get_latest_frame():
    frame_path = os.path.join(OUTPUT_DIR, "frame.jpg")
    if os.path.exists(frame_path):
        frame = cv2.imread(frame_path)
        return frame
    return None

# Webots controller logic
def run_webots_controller():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Initialize camera
    camera = robot.getDevice("receiver")
    camera.enable(TIME_STEP)

    while robot.step(TIME_STEP) != -1:
        # Capture camera image
        image = camera.getImage()
        np_image = np.frombuffer(image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        _, buffer = cv2.imencode('.jpg', np_image)

        # Save the frame to a file
        with open(os.path.join(OUTPUT_DIR, "frame.jpg"), "wb") as f:
            f.write(buffer)

        # Check for new commands
        if os.path.exists(COMMAND_FILE):
            with open(COMMAND_FILE, "r") as f:
                command = f.read().strip()
            os.remove(COMMAND_FILE)

            # Execute command
            if command == "move forward":
                # Set wheel velocities for moving forward
                robot.getMotor('left_wheel').setVelocity(1.0)
                robot.getMotor('right_wheel').setVelocity(1.0)
            elif command == "turn left":
                # Set wheel velocities for turning left
                robot.getMotor('left_wheel').setVelocity(-1.0)
                robot.getMotor('right_wheel').setVelocity(1.0)
            elif command == "turn right":
                # Set wheel velocities for turning right
                robot.getMotor('left_wheel').setVelocity(1.0)
                robot.getMotor('right_wheel').setVelocity(-1.0)
            elif command == "stop":
                # Set wheel velocities for stopping
                robot.getMotor('left_wheel').setVelocity(0)
                robot.getMotor('right_wheel').setVelocity(0)

# Start the Webots simulation as a subprocess
# webots_process = subprocess.Popen(['webots', '--mode=fast', 'path/to/your/world.wbt'])

# Start the Webots controller in a separate thread
controller_thread = Thread(target=run_webots_controller)
controller_thread.start()

# Streamlit main loop
try:
    while True:
        frame = get_latest_frame()

        if frame is not None:
            st.image(frame, channels="BGR")

        time.sleep(0.1)  # Adjust the sleep time as needed

except Exception as e:
    st.error(f"An error occurred: {e}")

finally:
    # webots_process.terminate()
    controller_thread.join()
