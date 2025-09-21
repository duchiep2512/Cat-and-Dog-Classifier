"""
Cat vs Dog Classifier - Clean Version
Author: duchiep2512
Professional ML Application
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

# Configure page
st.set_page_config(
    page_title="Cat vs Dog AI Classifier",
    page_icon="🐾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-success {
        background: linear-gradient(90deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-info {
        background: linear-gradient(90deg, #3498db, #85c1e9);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    try:
        # Check if model file exists
        model_path = 'fine_tuned_best.h5'
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found")
            st.info("Please ensure the model file is uploaded to your deployment service")
            return None
        
        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("AI Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please check if the model file is compatible with this TensorFlow version")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (224x224)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

def predict_image(model, image_array):
    """Make prediction on preprocessed image"""
    try:
        # Make prediction
        prediction = model.predict(image_array, verbose=0)
        probability = float(prediction[0][0])
        
        # Determine class and confidence
        if probability > 0.5:
            predicted_class = "Dog"
            confidence = probability
        else:
            predicted_class = "Cat"
            confidence = 1 - probability
        
        return predicted_class, confidence
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Cat vs Dog AI Classifier</h1>
        <p>Advanced Machine Learning Application</p>
        <p>Professional ML Portfolio Project</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### AI Model Info")
        st.info("""
        **Cat vs Dog Classifier**
        - Deep Learning CNN
        - Accuracy: ~87%
        - Input: 224x224 RGB images
        - Classes: Cat, Dog
        """)
        
        st.markdown("### Developer")
        st.markdown("""
        **Professional ML Developer**
        - ML & AI Specialist
        - Data Science Projects
        - Computer Vision Expert
        """)
        
        st.markdown("### Portfolio Projects")
        st.markdown("""
        - **Chat_box_LLM**: AI Chatbot
        - **Fake_News_Detection**: NLP Classification
        - **House_price_prediction**: Regression Analysis
        - **Health_Analytics**: Medical Data Analysis
        - **THPT Score Analysis**: Educational Analytics
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Image URL", "Examples"])
    
    # Tab 1: Upload Image
    with tab1:
        st.markdown("### Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload a clear image of a cat or dog for AI classification"
        )
        
        if uploaded_file is not None:
            try:
                # Load and display image
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### Your Image")
                    st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
                    
                    # Image info
                    st.markdown(f"""
                    **Image Details:**
                    - Size: {image.size[0]} × {image.size[1]} pixels
                    - Format: {image.format}
                    - Mode: {image.mode}
                    """)
                
                with col2:
                    st.markdown("#### AI Analysis")
                    
                    if st.button("Classify with AI", type="primary", use_container_width=True):
                        with st.spinner("AI is analyzing your image..."):
                            # Preprocess image
                            image_array = preprocess_image(image)
                            
                            if image_array is not None:
                                # Make prediction
                                predicted_class, confidence = predict_image(model, image_array)
                                
                                if predicted_class is not None:
                                    # Display results
                                    if confidence >= 0.8:
                                        result_class = "result-success"
                                        confidence_text = "Very High Confidence"
                                    elif confidence >= 0.6:
                                        result_class = "result-info"
                                        confidence_text = "Good Confidence"
                                    else:
                                        result_class = "result-info"
                                        confidence_text = "Moderate Confidence"
                                    
                                    st.markdown(f"""
                                    <div class="{result_class}">
                                        <h2>{predicted_class}</h2>
                                        <h3>{confidence:.1%} Confidence</h3>
                                        <p>{confidence_text}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Progress bar
                                    st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                                    
                                    # Detailed metrics
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Prediction", predicted_class)
                                    with col_b:
                                        st.metric("Confidence", f"{confidence:.1%}")
                                    with col_c:
                                        st.metric("Category", confidence_text)
                                    
                                    # Interpretation
                                    if confidence >= 0.8:
                                        st.success("Excellent prediction! The AI is very confident.")
                                    elif confidence >= 0.6:
                                        st.info("Good prediction with reliable confidence.")
                                    else:
                                        st.warning("Moderate confidence. Try a clearer image.")
                            
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    # Tab 2: Image URL
    with tab2:
        st.markdown("### Analyze Image from URL")
        
        image_url = st.text_input(
            "Enter image URL:",
            placeholder="https://example.com/cat-or-dog-image.jpg",
            help="Paste a direct link to an image file"
        )
        
        if image_url and st.button("Classify from URL", type="primary"):
            try:
                with st.spinner("Downloading image..."):
                    response = requests.get(image_url, timeout=15, headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; Cat-Dog-Classifier/1.0)'
                    })
                    response.raise_for_status()
                    
                    image = Image.open(BytesIO(response.content))
                
                # Display and analyze
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Downloaded Image", use_column_width=True)
                
                with col2:
                    with st.spinner("AI analyzing..."):
                        image_array = preprocess_image(image)
                        if image_array is not None:
                            predicted_class, confidence = predict_image(model, image_array)
                            
                            if predicted_class is not None:
                                st.success(f"**{predicted_class}** ({confidence:.1%} confidence)")
                                st.progress(confidence)
            
            except requests.exceptions.RequestException as e:
                st.error(f"Could not download image: {e}")
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    # Tab 3: Examples
    with tab3:
        st.markdown("### Try Example Images")
        st.markdown("Click any example below to test the AI classifier:")
        
        examples = [
            {
                "name": "Orange Tabby Cat",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/400px-Kittyply_edit1.jpg",
                "expected": "Cat"
            },
            {
                "name": "Golden Retriever",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Golden_Retriever_1_year-Edited.jpg/400px-Golden_Retriever_1_year-Edited.jpg",
                "expected": "Dog"
            },
            {
                "name": "British Shorthair",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Britishblue.jpg/400px-Britishblue.jpg",
                "expected": "Cat"
            }
        ]
        
        cols = st.columns(len(examples))
        
        for i, (col, example) in enumerate(zip(cols, examples)):
            with col:
                if st.button(f"Test {example['name']}", key=f"example_{i}"):
                    try:
                        with st.spinner(f"Testing {example['name']}..."):
                            response = requests.get(example['url'], timeout=10)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content))
                            
                            image_array = preprocess_image(image)
                            if image_array is not None:
                                predicted_class, confidence = predict_image(model, image_array)
                                
                                if predicted_class is not None:
                                    correct = predicted_class == example['expected']
                                    status = "Correct" if correct else "Incorrect"
                                    
                                    if correct:
                                        st.success(f"{status}: **{predicted_class}** ({confidence:.1%})")
                                    else:
                                        st.error(f"{status}: Got **{predicted_class}** | Expected {example['expected']}")
                                    
                                    with st.expander(f"View {example['name']}"):
                                        st.image(image, caption=f"{example['name']} | Result: {predicted_class}")
                    
                    except Exception as e:
                        st.error(f"Error with {example['name']}: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;">
        <strong>Professional ML Application Portfolio</strong><br>
        <strong>Tech Stack:</strong> TensorFlow • Streamlit • Computer Vision • Deep Learning<br>
        <strong>Deployed on:</strong> Cloud Infrastructure
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()