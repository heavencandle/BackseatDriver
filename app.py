import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel
import carla

st.title("Interactive Autonomous Vehicle Control and Simulation with Natural Language Commands")
st.write("Control and simulate an autonomous vehicle in a virtual environment using natural language commands.")

# Load models
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
nlp_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Connect to CARLA simulator
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

def send_command(command):
    # Translate the command to vehicle control actions and send them to CARLA
    if "left" in command:
        vehicle.apply_control(carla.VehicleControl(steer=-0.3))
    elif "right" in command:
        vehicle.apply_control(carla.VehicleControl(steer=0.3))
    elif "forward" in command:
        vehicle.apply_control(carla.VehicleControl(throttle=0.5))
    elif "stop" in command:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

def preprocess_frame(frame):
    img = Image.fromarray(frame)
    return img

def detect_objects(frame):
    results = model(frame)
    return results.pandas().xyxy[0]  # Bounding boxes and labels

def generate_scene_description(image):
    inputs = processor(text=["a photo of something"], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)  # Get probabilities
    return probs

def interpret_command(command):
    inputs = tokenizer(command, return_tensors="pt")
    outputs = nlp_model.generate(inputs)
    return tokenizer.decode(outputs[0])

def generate_actions(scene_description, command):
    actions = interpret_command(command)
    return actions

def display_simulation(frame, actions):
    send_command(actions)
    st.image(frame, channels="BGR")
    st.write(f"Actions: {actions}")

if st.button('Start Simulation'):
    cap = cv2.VideoCapture(0)  # Simulated video feed from virtual environment
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        scene_description = generate_scene_description(frame)
        st.write(f"Scene Description: {scene_description}")
        command = st.text_input("Enter a command for the vehicle:")
        if command:
            actions = generate_actions(scene_description, command)
            display_simulation(frame, actions)
    cap.release()

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    scene_description = generate_scene_description(image)
    st.write(f"Scene Description: {scene_description}")
    command = st.text_input("Enter a command for the vehicle:")
    if command:
        actions = generate_actions(scene_description, command)
        display_simulation(image, actions)
