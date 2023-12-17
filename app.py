import streamlit as st
import gdown
import torch

# Function to download the model
def download_model():
    file_id = '1IjMM3B8Wk0g1Y-1vsd6M_L6_JBw2qYPq'
    output_path = '/content'  # Adjust the output path as needed
    download_link = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_link, output_path, quiet=False)
    return output_path

# Download the model
model_path = download_model()

# Load the model
model = torch.load(model_path)

# Streamlit app
def main():
    st.title('Your Streamlit App Title')

    # Your Streamlit app content goes here

    # Example: Display a prediction using the loaded model
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        # Make predictions using your model
        prediction = model.predict(uploaded_file)

        # Display the prediction
        st.write(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()
