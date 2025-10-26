import streamlit as st
import mlflow  
import pandas as pd
import numpy as np

# --- 1. Load Model and Data ---

st.set_page_config(
    page_title="India Crop Production Predictor",
    page_icon="🌾",
    layout="wide"
)

@st.cache_resource
def load_model(model_name, stage_or_alias):
    """Loads the model from the MLflow Model Registry."""
    try:
        # --- THIS IS THE FIX ---
        # The syntax for an ALIAS uses '@' instead of '/'
        model_uri = f"models:/{model_name}@{stage_or_alias}" 
        # --- END OF FIX ---
        
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the model from MLflow: {e}")
        st.info("Please make sure the MLflow server is running (run 'mlflow ui') and the model has an ALIAS (e.g., 'Production').")
        return None

@st.cache_data
def load_data(data_path):
    """Loads the dataset for populating dropdowns."""
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {data_path}")
        st.info(f"Please make sure '{data_path}' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# --- Load the artifacts ---
# We are loading the 'Production' ALIAS
model = load_model(model_name='crop', stage_or_alias='Production')
df = load_data('crop_production.csv') 

# --- 2. Streamlit UI Layout ---

st.title("🌾 India Crop Production Predictor")
st.markdown("Use the sidebar to input parameters and get a production forecast.")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

if df is not None:
    # --- Create Dropdowns ---
    
    states = sorted(df['State_Name'].unique())
    seasons = sorted(df['Season'].unique())
    crops = sorted(df['Crop'].unique())
    
    selected_state = st.sidebar.selectbox("Select State", states)
    
    if selected_state:
        districts = sorted(df[df['State_Name'] == selected_state]['District_Name'].unique())
        selected_district = st.sidebar.selectbox("Select District", districts)
    else:
        selected_district = st.sidebar.selectbox("Select District", [])

    selected_season = st.sidebar.selectbox("Select Season", seasons)
    selected_crop = st.sidebar.selectbox("Select Crop", crops)
    
    selected_year = st.sidebar.number_input("Enter Crop Year", 
                                            min_value=1997, 
                                            max_value=2030, 
                                            value=2015)
                                            
    selected_area = st.sidebar.number_input("Enter Area (in Hectares)", 
                                            min_value=0.01, 
                                            value=100.0, 
                                            step=1.0)
    
    # --- 3. Prediction Logic ---
    
    if st.sidebar.button("Predict Production"):
        if model is not None and selected_district:
            try:
                input_data = pd.DataFrame({
                    'State_Name': [selected_state],
                    'District_Name': [selected_district],
                    'Crop_Year': [selected_year],
                    'Season': [selected_season],
                    'Crop': [selected_crop],
                    'Area': [selected_area]
                })
                
                st.subheader("Model Input:")
                st.dataframe(input_data)
                
                prediction = model.predict(input_data)
                
                st.subheader("Prediction Result:")
                st.metric(
                    label="Predicted Crop Production",
                    value=f"{prediction[0]:,.2f} units"
                )
                st.info("Note: 'units' refers to the same unit of production (e.g., tons, kgs) as in the original dataset.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            if model is None:
                st.error("Model is not loaded. Cannot make predictions.")
            if not selected_district:
                st.warning("Please select a district.")
else:
    st.error("Application cannot start. Please check file paths for model and data.")

st.sidebar.markdown("---")
st.sidebar.markdown("This app uses a pre-trained XGBoost model (served via MLflow) to predict crop production.")

