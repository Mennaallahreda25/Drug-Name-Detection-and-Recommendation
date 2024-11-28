#Install Required Libraries
#!pip install ultralytics transformers datasets accelerate torch pandas matplotlib seaborn
#-----------------------------------------------------------------------------------------------

#Drug Name Detection and Recommendation Code
import os
import yaml
import glob
import random
import json
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# YOLO Drug Name Detection Configuration
def train_yolo_model():
    # YOLO model setup
    model = YOLO("yolov8n.pt")  # Load pretrained YOLOv8n model

    # Dataset configuration
    config = {
        "path": "./drug-name-detection-dataset",
        "train": "./drug-name-detection-dataset/train",
        "val": "./drug-name-detection-dataset/valid",
        "test": "./drug-name-detection-dataset/test",
        "nc": 1,  # Number of classes
        "names": ["drug-name"],  # Class name
    }

    # Save dataset config
    with open("data.yaml", "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    # Train the YOLO model
    results = model.train(data="data.yaml", epochs=50, save_period=10, seed=seed, name="yolov8n")
    print("Training completed.")
    return model

def detect_drug_names(model, test_image_paths):
    # Predict drug names from test images
    results = model.predict(test_image_paths)
    detected_drugs = []

    for i, r in enumerate(results):
        drug_names = [res['label'] for res in r.boxes.data.cpu().numpy()]
        detected_drugs.append({
            'image_id': test_image_paths[i],
            'detected_drug_names': drug_names
        })

    # Save detected drugs
    with open("detected_drugs.json", "w") as f:
        json.dump(detected_drugs, f)

    return detected_drugs

# Medicine Recommendation System
def load_medicine_data():
    # Load medical dataset
    df = pd.read_csv('./drugs_side_effects_drugs_com.csv')
    
    # Create input and output columns for the recommendation system
    def create_input(row):
        return f"Patient Info:\n- Medical Condition: {row['medical_condition']}\n"

    def create_output(row):
        return (f"Considering your medical condition of {row['medical_condition']}, "
                f"you might want to take drugs like {row['drug_name']}. "
                f"This might accompany side effects such as {row['side_effects']}, "
                f"so you should be aware of this when taking it.")

    df['input'] = df.apply(create_input, axis=1)
    df['output'] = df.apply(create_output, axis=1)
    df['text'] = df.apply(lambda row: f"input: {row['input']}\noutput: {row['output']}", axis=1)
    
    # Remove missing values and shuffle the data
    df.dropna(subset=['input', 'output', 'text'], inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def load_model_and_tokenizer(model_name):
    # Load pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=1024)
    return model, tokenizer

def get_medicine_recommendation(medical_condition, df):
    # Match medical condition in dataset
    matching_rows = df[df['medical_condition'].str.lower() == medical_condition.lower()]
    
    if matching_rows.empty:
        return f"We do not have information about this medical condition."

    # Return the first matched row's output
    row = matching_rows.iloc[0]
    return row['output']

# Main Pipeline
def main_pipeline():
    # Train YOLO model
    print("Training YOLO model...")
    model = train_yolo_model()

    # Detect drugs in test images
    test_image_paths = glob.glob("./drug-name-detection-dataset/test/images/*")
    print("Detecting drugs in test images...")
    detected_drugs = detect_drug_names(model, test_image_paths)

    # Load medicine recommendation data
    print("Loading medicine recommendation data...")
    df = load_medicine_data()

    # Load pre-trained recommendation model
    print("Loading recommendation model...")
    model_name = "gpt2"  # Replace with your model path if using a custom model
    recommendation_model, tokenizer = load_model_and_tokenizer(model_name)

    # Process each detected drug and generate recommendations
    print("\nProcessing detected drugs...")
    for item in detected_drugs:
        print(f"Image: {item['image_id']}")
        for drug_name in item['detected_drug_names']:
            recommendation = get_medicine_recommendation(drug_name, df)
            print(f"Detected Drug: {drug_name}")
            print(f"Recommendation: {recommendation}\n")

if __name__ == "__main__":
    main_pipeline()

