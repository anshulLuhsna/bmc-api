from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS  # Import CORS
from sklearn.preprocessing import LabelEncoder
import numpy as np
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)
# Load the trained model pipeline
with open('random_forest_classifier.pkl', 'rb') as f:
    pipeline = pickle.load(f)

def process_data(file):
    stats = {}
    
    # Load the CSV file
    data = pd.read_csv(file)
    original_shape = data.shape
    stats['original_shape'] = original_shape
    
    # Step 1: Drop columns that are completely null
    columns_before = data.columns.tolist()
    data_cleaned = data.dropna(axis=1, how='all')
    dropped_columns_all_null = list(set(columns_before) - set(data_cleaned.columns))
    stats['dropped_columns_all_null'] = dropped_columns_all_null
    stats['remaining_columns_after_dropping_all_null'] = data_cleaned.shape[1]

    # Step 2: Drop columns with more than 50% null values
    threshold = 0.5 * len(data_cleaned)
    columns_before = data_cleaned.columns.tolist()
    data_cleaned = data_cleaned.dropna(axis=1, thresh=threshold)
    dropped_columns_more_than_50_null = list(set(columns_before) - set(data_cleaned.columns))
    stats['dropped_columns_more_than_50_null'] = dropped_columns_more_than_50_null
    stats['remaining_columns_after_50_percent_null'] = data_cleaned.shape[1]

    # Step 3: Drop rows that contain at least one null value
    rows_before = data_cleaned.shape[0]
    data_cleaned = data_cleaned.dropna()
    rows_dropped = rows_before - data_cleaned.shape[0]
    stats['rows_dropped_due_to_nulls'] = rows_dropped
    stats['remaining_rows_after_dropping_nulls'] = data_cleaned.shape[0]

    # Step 4: Identify numeric and non-numeric columns
    numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_columns = data_cleaned.select_dtypes(exclude=['float64', 'int64']).columns
    stats['numeric_columns'] = numeric_columns.tolist()
    stats['non_numeric_columns'] = non_numeric_columns.tolist()

    # Step 5: Calculate correlations for numeric columns
    if 'ReconciliationRuleId' in numeric_columns:
        numeric_columns = numeric_columns.drop('ReconciliationRuleId')
    data_numeric = data_cleaned[numeric_columns]
    corr_numeric = data_numeric.corrwith(data_cleaned['ReconciliationRuleId']).abs()

    # Step 6: Encode non-numeric columns and calculate correlations
    label_encoder = LabelEncoder()
    encoded_data = data_cleaned.copy()
    encoding_info = {}
    for col in non_numeric_columns:
        encoded_data[col] = label_encoder.fit_transform(encoded_data[col].astype(str))
        encoding_info[col] = {'unique_values': len(data_cleaned[col].unique())}
    stats['encoding_info'] = encoding_info
    
    # Correlation for non-numeric columns (encoded)
    encoded_corr = encoded_data[non_numeric_columns].corrwith(encoded_data['ReconciliationRuleId']).abs()

    # Step 7: Combine numeric and encoded non-numeric correlations
    combined_corr = pd.concat([corr_numeric, encoded_corr]).sort_values(ascending=False)

    # Convert NaN, inf, and -inf values to None (null) in all statistics
    combined_corr = combined_corr.replace([np.nan, np.inf, -np.inf], None)

    # Add the correlations to the stats
    stats['correlations'] = combined_corr.to_dict()

    return stats


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Convert the input JSON to a DataFrame
    input_df = pd.DataFrame(data, index=[0])

    # Ensure the input has the correct numeric and categorical columns
    required_columns = ['createdate', 'numberofslots', 'modifieddate', 'totalphysicalmemory',
                        'requestid', 'shortdescription', 'partitionid', 'workgroup',
                        'reconciliationidentity', 'datasetid', 'versionnumber', 
                        'manufacturername', 'lastrejobrunid', 'domain']

    if set(required_columns).issubset(input_df.columns):
        # Get predictions and probabilities
        prediction = pipeline.predict(input_df)
        probabilities = pipeline.predict_proba(input_df)

        # Assuming you have class labels available
        class_labels = ['Rule 100', 'Rule 110', 'Rule 120', 'Rule 130']  # Update this based on your model

        # Prepare the response with class labels and their probabilities
        response = {
            'prediction': prediction.tolist(),
            'probabilities': probabilities.tolist(),
            'class_labels': class_labels  # Adding class labels to the response
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'Input data must have the required columns'}), 400

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Process the file and get detailed stats and correlations
    stats = process_data(file)
    print(stats)
    # Return the result as a JSON response
    return jsonify({"stats": stats})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
