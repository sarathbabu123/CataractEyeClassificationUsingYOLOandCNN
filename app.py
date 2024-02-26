# Import necessary libraries
import os
from flask import Flask, flash, redirect, request, url_for, render_template, get_flashed_messages,Response
from werkzeug.utils import secure_filename
from prediction import prediction,yolo  # Import prediction function from prediction module
import cv2 as cv

# Define the upload folder and allowed extensions for uploaded files
UPLOAD_FOLDER = "./static/uploads/"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize the Flask application
app = Flask(__name__)
app.debug = True  # Enable debug mode
app.secret_key = "asarath"  # Set the secret key for the application

# Configure the upload folder
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the route for the upload page
@app.route("/", methods=["GET", "POST"])
def upload_file():
    img = None 
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = 'uploads/' + filename  # Dynamic image path
            print(img)
            pred = prediction(filepath)  # Get prediction for the uploaded image
            if pred == "Cataract":
                flash("It is a Cataract Eye")
            else:
                flash("It is a Normal Eye")
            # Render the index page with the uploaded image and prediction result
            return render_template("index.html", img=img, messages=get_flashed_messages())  

    # Render the index page
    return render_template("index.html", img=img, messages=get_flashed_messages())  

# Define the route for the video page
@app.route("/video")
def video():
    # Return a response with the video feed from the yolo function
    return Response(yolo(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the application
if __name__ == "__main__":
    app.run()