import streamlit as st
import os
from PIL import Image
import numpy as np
import cv2
from skimage import morphology
import pandas as pd

# Function to normalize hue
def normalize_hue(h):
    return int(h * 255 / 360)

# Function to process a single image
def process_image(image_path, disease_hue, chlorosis_hue_range):
    image = Image.open(image_path)
    image_np = np.array(image)
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    disease_hue = normalize_hue(disease_hue)
    chlorosis_hue_min, chlorosis_hue_max = map(normalize_hue, chlorosis_hue_range)

    # Create binary masks
    leaf_mask = cv2.inRange(hsv_image, np.array([0, 0, 0], dtype=np.uint8), np.array([90, 255, 255], dtype=np.uint8))
    leaf_mask = morphology.remove_small_objects(leaf_mask.astype(bool), 500)
    leaf_mask = morphology.remove_small_holes(leaf_mask, 500)

    disease_mask = cv2.inRange(hsv_image, np.array([0, 0, 0], dtype=np.uint8), np.array([disease_hue, 255, 255], dtype=np.uint8))
    disease_mask = morphology.remove_small_objects(disease_mask.astype(bool), 300)
    disease_mask = morphology.remove_small_holes(disease_mask, 300)

    chlorosis_mask = cv2.inRange(hsv_image, np.array([disease_hue, 0, 0], dtype=np.uint8), np.array([chlorosis_hue_max, 255, 255], dtype=np.uint8))
    chlorosis_mask = morphology.remove_small_objects(chlorosis_mask.astype(bool), 300)
    chlorosis_mask = morphology.remove_small_holes(chlorosis_mask, 300)

    # Calculate areas
    leaf_area = np.sum(leaf_mask)
    disease_area = np.sum(disease_mask)
    chlorosis_area = np.sum(chlorosis_mask)
    healthy_area = leaf_area - disease_area - chlorosis_area
    
    return {
        "leaf_area": leaf_area,
        "healthy_area": healthy_area,
        "disease_area": disease_area,
        "chlorosis_area": chlorosis_area,
    }

# Streamlit app
st.title("Leaf Health Analyzer")

# Upload images
uploaded_files = st.file_uploader(
    "Upload Images (TIF, PNG, JPG, JPEG)", 
    type=["tif", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

# User inputs for hue values
st.sidebar.header("Hue Configuration")
disease_hue = st.sidebar.slider("Disease Hue", 0, 360, 30)
chlorosis_hue_max = st.sidebar.slider("Chlorosis Hue Max", 0, 360, 43)

if st.button("Process Images"):
    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            # Save uploaded file to a temporary directory
            temp_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Process the image
            st.info(f"Processing {uploaded_file.name}...")
            result = process_image(temp_path, disease_hue, (chlorosis_hue_min, chlorosis_hue_max))
            result["filename"] = uploaded_file.name
            results.append(result)

        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Display results
        st.success("Processing complete!")
        st.write("Results:")
        st.dataframe(df)

        # Provide CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="leaf_health_analysis.csv",
            mime="text/csv"
        )
    else:
        st.error("Please upload at least one image.")
