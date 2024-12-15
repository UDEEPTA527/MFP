import numpy as np
import pickle
import streamlit as st
import warnings

# Load the saved models
loaded_machine_failure_model = pickle.load(open("Machine_failure.sav", 'rb'))
loaded_failure_type_model = pickle.load(open("failure_type_model.sav", 'rb'))

def predict_machine_failure(input_data):
    """
    Predict machine failure and failure type.
    """
    try:
        # Convert input data to a NumPy array
        input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)
        
        # Predict machine failure
        machine_failure = loaded_machine_failure_model.predict(input_data_as_numpy_array)
        
        # Predict failure type
        failure_type = loaded_failure_type_model.predict(input_data_as_numpy_array)
        
        return machine_failure[0], failure_type[0], None  # Return predictions and no error
    except Exception as e:
        return None, None, str(e)  # Return None and error message

def validate_input(input_data):
    """
    Validate user inputs.
    """
    error_messages = {}
    input_field_names = [
        'Type', 'Air Temperature', 'Process Temperature', 
        'Rotational Speed', 'Torque', 'Tool Wear'
    ]
    
    for i, val in enumerate(input_data):
        field_name = input_field_names[i]
        if not val:
            error_messages[field_name] = f"{field_name} is missing."
        else:
            try:
                # Check if the input can be converted to float
                float_val = float(val)
                if float_val < 0:
                    error_messages[field_name] = f"{field_name} cannot be negative."
            except ValueError:
                error_messages[field_name] = f"Invalid value for {field_name}. Please enter a valid number."
    
    return error_messages

def main():
    """
    Streamlit app for Machine Failure Prediction.
    """
    st.title("Machine Failure Prediction Web App")
    
    # Input fields
    col1, col2 = st.columns(2)
    input_field_names = [
        'Type', 'Air Temperature', 'Process Temperature', 
        'Rotational Speed', 'Torque', 'Tool Wear'
    ]
    input_data = {}
    
    for i, field_name in enumerate(input_field_names):
        with col1 if i % 2 == 0 else col2:
            input_data[field_name] = st.text_input(f"{i+1}. {field_name} value")
    
    # Placeholders for results
    machine_failure_result = ""
    failure_type_result = ""
    prediction_error_message = ""
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    
    # Prediction button
    if st.button("Predict Machine Failure"):
        # Validate inputs
        error_messages = validate_input(list(input_data.values()))
        
        if not error_messages:
            input_values = list(input_data.values())
            machine_failure, failure_type, prediction_error = predict_machine_failure(input_values)
            
            if prediction_error:
                prediction_error_message = f"Error during prediction: {prediction_error}"
            else:
                # Interpret and display results
                machine_failure_result = (
                    "Machine Failure: Yes" if machine_failure == 1 else "Machine Failure: No"
                )
                failure_type_result = f"Failure Type: {failure_type}"
    
    # Display error messages
    for field_name, error_message in error_messages.items():
        st.error(f"{field_name}: {error_message}")
    
    # Display prediction results
    if prediction_error_message:
        st.error(prediction_error_message)
    else:
        st.success(machine_failure_result)
        st.success(failure_type_result)

if __name__ == "__main__":
    main()
