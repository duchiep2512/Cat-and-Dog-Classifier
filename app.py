"""
Simple Cat vs Dog Classifier
Author: duchiep2512
Clean and simple Streamlit app for existing model
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Page config
st.set_page_config(
    page_title="🐱🐕 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .cat-result {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
    }
    .dog-result {
        background-color: #d1ecf1;
        border: 1px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('fine_tuned_best.h5')
        st.success(" Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f" Error loading model: {e}")
        st.info("Make sure 'fine_tuned_best.h5' is in the same folder")
        return None

def preprocess_image(image):
    """Simple image preprocessing"""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_image(model, image_array):
    """Make prediction"""
    prediction = model.predict(image_array, verbose=0)
    probability = float(prediction[0][0])
    
    if probability > 0.5:
        return "Dog", probability, "🐕"
    else:
        return "Cat", 1 - probability, "🐱"

def main():
    """Main app"""
    
    # Header
    st.markdown('<h1 class="main-header">🐾 Cat vs Dog Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Upload an image to classify cats and dogs</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### ℹ About")
        st.info("""
        **Simple Cat vs Dog Classifier**
        - Upload image or paste URL
        - AI will classify as Cat or Dog
        - Shows confidence percentage
        """)
       
    
    # Main interface - tabs
    tab1, tab2 = st.tabs([" Upload", " URL"])
    
    # Upload tab
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image", 
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Show image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Your Image", use_column_width=True)
            
            with col2:
                if st.button(" Classify", type="primary"):
                    with st.spinner("Analyzing..."):
                        # Preprocess and predict
                        image_array = preprocess_image(image)
                        predicted_class, confidence, emoji = predict_image(model, image_array)
                        
                        # Show result
                        result_class = "cat-result" if predicted_class == "Cat" else "dog-result"
                        
                        st.markdown(f"""
                        <div class="result-box {result_class}">
                            <h2>{emoji} {predicted_class}</h2>
                            <h3>{confidence:.1%} confidence</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.progress(confidence)
                        
                        # Simple interpretation
                        if confidence > 0.8:
                            st.success("Very confident!")
                        elif confidence > 0.6:
                            st.info("Good confidence")
                        else:
                            st.warning("Low confidence")
    
    # URL tab
    with tab2:
        image_url = st.text_input(
            "Image URL:", 
            placeholder="https://example.com/image.jpg"
        )
        
        if image_url and st.button("🔍 Classify from URL", type="primary"):
            try:
                with st.spinner("Downloading image..."):
                    response = requests.get(image_url, timeout=10)
                    image = Image.open(BytesIO(response.content))
                
                # Show and classify
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Downloaded Image", use_column_width=True)
                
                with col2:
                    with st.spinner("Analyzing..."):
                        image_array = preprocess_image(image)
                        predicted_class, confidence, emoji = predict_image(model, image_array)
                        
                        result_class = "cat-result" if predicted_class == "Cat" else "dog-result"
                        
                        st.markdown(f"""
                        <div class="result-box {result_class}">
                            <h2>{emoji} {predicted_class}</h2>
                            <h3>{confidence:.1%} confidence</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(confidence)
                        
                        if confidence > 0.8:
                            st.success("Very confident!")
                        elif confidence > 0.6:
                            st.info("Good confidence")
                        else:
                            st.warning("Low confidence")
            
            except Exception as e:
                st.error(f"Error: {e}")
      
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;"> Built with TensorFlow & Streamlit |  duchiep2512</p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()