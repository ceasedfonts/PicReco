from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch
import streamlit as st

st.markdown("<h2 style='text-align: center; color: white;'>This tool is using Google's ViT model to classify the uploaded image</h2>", unsafe_allow_html = True)
image = st.file_uploader("Choose your image", type = ['jpg']) # type = ['jpg'] is used to filter the file types

if image is not None:
    image = Image.open(image)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224') # ViTForImageClassification
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224') # ViTImageProcessor

    inputs = processor(images = image, return_tensors = "pt") # return_tensors = "pt" is used to return PyTorch tensors
    outputs = model(**inputs) # **inputs is used to unpack the dictionary into keyword arguments
    logits = outputs.logits # logits is a tensor of shape (batch_size, number_of_labels) where each value is a probability score between 0 and 1, and the sum of all values is 1
    
    predicted_class_idx = logits.argmax(-1).item() # predicted_class_idx is the index in a list of labels

    st.markdown("<h2 style='text-align: center; color: white;'>Classification</h2>", unsafe_allow_html = True)
    st.header(model.config.id2label[predicted_class_idx])


