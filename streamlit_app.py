import numpy as np
import pickle
import streamlit as st
import warnings

# Load the saved models
loaded_model1 = pickle.load(open("Heating_load_model_rf.sav", 'rb'))
loaded_model2 = pickle.load(open("Cooling_load_model_rf.sav", 'rb'))

def building_load_prediction(input_data):
    try:
        # Convert input_data to float and reshape
        input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    
        HeatingLoad = loaded_model1.predict(input_data_as_numpy_array)
        CoolingLoad = loaded_model2.predict(input_data_as_numpy_array)
    
        return HeatingLoad[0], CoolingLoad[0], None  # Return results and no error message
    except Exception as e:
        return None, None, str(e)  # Return None for results and the error message

def validate_input(input_data):
    error_messages = {}
    
    # Define input field names for reference
    input_field_names = [
        'RelativeCompactness', 'SurfaceArea', 'WallArea', 'RoofArea',
        'OverallHeight', 'Orientation', 'GlazingArea', 'GlazingAreaDistribution'
    ]
    
    for i, val in enumerate(input_data):
        field_name = input_field_names[i]
        
        if not val:
            error_messages[field_name] = f"{field_name} is missing."
        else:
            try:
                # Attempt to convert input value to float
                float_val = float(val)
                if float_val < 0:
                    error_messages[field_name] = f"{field_name} cannot be negative."
            except ValueError:
                error_messages[field_name] = f"Invalid value for {field_name}. Please enter a valid number."
    
    return error_messages

def main():
    st.title('Energy Efficient Building Load Calculation Web App')
    
    # Collect input data from the user
    
    # Getting the input data from the user
    col1, col2 = st.columns(2)
    
    input_field_names = [
        'RelativeCompactness', 'SurfaceArea', 'WallArea', 'RoofArea',
        'OverallHeight', 'Orientation', 'GlazingArea', 'GlazingAreaDistribution'
    ]
    
    input_data = {}
    
    for i, field_name in enumerate(input_field_names):
        with col1 if i % 2 == 0 else col2:
            input_data[field_name] = st.text_input(f"{i+1}. {field_name} value")
    
    # Create placeholders for displaying the results and error messages
    HeatingLoad_result = ""
    CoolingLoad_result = ""
    prediction_error_message = ""
    
    # Initialize error_messages with an empty dictionary
    error_messages = {}
    
    # Suppress the sklearn warning related to feature names
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    
    # Check if the user has clicked the prediction button
    if st.button('Estimate the Load'):
        error_messages = validate_input(list(input_data.values()))
        
        if not error_messages:
            input_values = list(input_data.values())
            heating_load, cooling_load, prediction_error = building_load_prediction(input_values)
            
            if prediction_error is not None:
                prediction_error_message = f"Error during prediction: {prediction_error}"
            else:
                HeatingLoad_result = f"Estimated Heating Load value is: {round(heating_load, 2)}"  # Display heating load
                CoolingLoad_result = f"Estimated Cooling Load value is: {round(cooling_load, 2)}"  # Display cooling load
    
    # Display the results or error messages in the Streamlit app
    for field_name, error_message in error_messages.items():
        st.error(f"{field_name}: {error_message}")
    
    if prediction_error_message:
        st.error(prediction_error_message)
    else:
        st.success(HeatingLoad_result)
        st.success(CoolingLoad_result)

if __name__ == '__main__':
    main()
