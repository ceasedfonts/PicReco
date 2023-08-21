from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch
import streamlit as st
import torch.nn.functional as nnf

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
    
    predicted_class_idx = logits.argmax(-1).item() # logits.argmax(-1).item() is the predicted label

    prob = nnf.softmax(logits, dim = 1) # softmax is used to convert the logits into probabilities
    top_p, top_class = prob.topk(3, dim = 1) # topk(3, dim = 1) is used to get the top 3 predictionsround(top_p[0][0].item(), 2)

    st.markdown("<h2 style='text-align: center; color: white;'>Top 3 Classifications and Probabilities</h2>", unsafe_allow_html = True)
    st.text(model.config.id2label[predicted_class_idx] + ' with probability ' + str(round(top_p[0][0].item(), 2)))
    st.text(model.config.id2label[top_class[0][1].item()] + ' with probability ' + str(round(top_p[0][1].item(), 2)))
    st.text(model.config.id2label[top_class[0][2].item()] + ' with probability ' + str(round(top_p[0][2].item(), 2)))
