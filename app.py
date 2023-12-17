import streamlit as st
from fastai.vision.all import *

# Load the best model
best_model_path = '/Users/yassinkissami/Downloads/RosesDataset 2/best_model.pth'  # Update with your actual path
loaded_model = load_learner(best_model_path)

# Streamlit app
def main():
    st.title("Rose Classification App")
    st.sidebar.title("Options")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Make predictions using the model
        prediction, _, probabilities = loaded_model.predict(PILImage.create(uploaded_file))
        
        # Display the prediction
        st.write(f"Prediction: {prediction}")
        st.write(f"Probabilities: {probabilities}")

if __name__ == "__main__":
    main()
