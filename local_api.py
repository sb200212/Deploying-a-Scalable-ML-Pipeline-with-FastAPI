import json
import requests

# Define the base URL for the API
BASE_URL = "http://127.0.0.1:8000"

print(" Testing GET Request to the Root Endpoint ")

try:
    # Send a GET request using the URL http://127.0.0.1:8000
    r_get = requests.get(BASE_URL)

    # Print the status code
    print(f"Status Code: {r_get.status_code}")
    
    # Print the welcome message from the JSON response
    print(f"Response Body: {r_get.json()}")

except requests.exceptions.ConnectionError as e:
    print(f"Connection Error: Could not connect to the API at {BASE_URL}.")
    print("Please ensure the FastAPI server is running with the command: uvicorn main:app --reload")
    exit() # Exit the script if the server isn't running


print("\n" + "="*50 + "\n")


print(" Testing POST Request to the Prediction Endpoint ")

# Data for a single prediction
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request using the data above
# Note: The endpoint is '/predict' based on the main.py code I provided.
# If you used a different path like '/data/', change the URL below.
r_post = requests.post(f"{BASE_URL}/predict", json=data)

# Print the status code
print(f"Status Code: {r_post.status_code}")

# Print the result from the JSON response
print(f"Response Body: {r_post.json()}")