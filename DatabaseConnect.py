from flask import Flask, render_template, request, send_file
from firebase_admin import credentials, db, initialize_app
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Initialize Firebase app
cred = credentials.Certificate('serviceAccountKey.json')
initialize_app(cred, {'databaseURL': 'https://resumebd-ad540-default-rtdb.europe-west1.firebasedatabase.app/'})

# Fetch data from Firebase
firebase_ref = db.reference('/1Ixr1_A0LUiN6g5gUdudQyR6MM7IlqufTzG1YpKrjjco/Feuille 1')
data = firebase_ref.get()

# Ensure that data is not None
if data is None:
    data = []

# Function to save the simple linear regression plot
def save_simple_linear_regression_plot(category):
    filtered_data = [entry_data for entry_data in data if entry_data and isinstance(entry_data, dict) and entry_data.get("Category") == category]

    if not filtered_data:
        print(f"No data found for category: {category}")
        return None

    x = np.array([entry_data.get("Age", "") for entry_data in filtered_data])
    y = np.array([entry_data.get("Years of Experience", "") for entry_data in filtered_data])

    model = LinearRegression().fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x, y, label='Actual Data')
    ax.plot(x, y_pred, color='red', label='Simple Linear Regression')
    ax.set_xlabel('Age')
    ax.set_ylabel('Years of Experience')
    ax.legend()

    # Save the plot as an image in memory
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    plt.close()

    return image_stream

# Function to save the multiple linear regression plot
def save_multiple_linear_regression_plot(category):
    filtered_data = [entry_data for entry_data in data if entry_data and isinstance(entry_data, dict) and entry_data.get("Category") == category]

    if not filtered_data:
        print(f"No data found for category: {category}")
        return None

    categories = [entry_data.get("Category", "") for entry_data in filtered_data]
    categories = np.array(categories).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    categories_encoded = encoder.fit_transform(categories)

    x = np.array(
        [[entry_data.get("Age", "")] + list(category_row) for entry_data, category_row in zip(filtered_data, categories_encoded)])
    y = np.array([entry_data.get("Years of Experience", "") for entry_data in filtered_data])

    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(y, y_pred, label='Multiple Linear Regression')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--', color='red', label='Perfect Prediction')
    ax.set_xlabel('Actual Years of Experience')
    ax.set_ylabel('Predicted Years of Experience')
    ax.legend()

    # Save the plot as an image in memory
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    plt.close()

    return image_stream

@app.route('/')
def index():
    categories = set(entry_data.get("Category", "") for entry_data in data if entry_data and isinstance(entry_data, dict))
    return render_template('index.html', categories=list(categories))

@app.route('/get_data', methods=['POST'])
def get_data():
    category = request.form['category']
    filtered_data = [entry_data for entry_data in data if entry_data and isinstance(entry_data, dict) and entry_data.get("Category") == category]

    # Save plots
    simple_linear_regression_plot = save_simple_linear_regression_plot(category)
    multiple_linear_regression_plot = save_multiple_linear_regression_plot(category)

    return render_template('table.html', data=filtered_data, category=category,
                           simple_linear_regression_plot=simple_linear_regression_plot,
                           multiple_linear_regression_plot=multiple_linear_regression_plot)

@app.route('/plot/<plot_type>/<category>')
def plot(plot_type, category):
    # Retrieve the saved plot image stream
    if plot_type == 'simple':
        image_stream = save_simple_linear_regression_plot(category)
    elif plot_type == 'multiple':
        image_stream = save_multiple_linear_regression_plot(category)
    else:
        return 'Invalid plot type'

    # Return the image stream as a response
    return send_file(image_stream, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
