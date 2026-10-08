import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("Photo Describer")
st.write("Upload a picture and the app will write a short description.")

@st.cache_resource
def load_photo_describer():
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

photo = st.file_uploader(
    "Choose a picture",
    type=["jpg", "jpeg", "png"]
)

if photo is not None:
    image = Image.open(photo).convert("RGB")
    st.image(image, caption="Your picture", use_container_width=True)

    if st.button("Describe this picture"):
        with st.spinner("Looking at the picture..."):
            describer = load_photo_describer()
            result = describer(image)
            st.write("**Description:**", result[0]["generated_text"])
