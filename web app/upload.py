from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

IMAGE_FOLDER = 'static/images'
VIDEO_FOLDER = 'static/videos'

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    message = None  # Initially, no message
    if request.method == 'POST':
        uploaded_file = request.files.get('media')

        if uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            ext = filename.rsplit('.', 1)[1].lower()
            
            # Check if the file already exists
            if ext in ALLOWED_IMAGE_EXTENSIONS:
                file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
                if os.path.exists(file_path):  # File already exists
                    message = "File already uploaded. Change the name or try again."
                else:
                    uploaded_file.save(file_path)
                    message = f"Image uploaded to: {url_for('static', filename='images/' + filename)}"
            elif ext in ALLOWED_VIDEO_EXTENSIONS:
                file_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
                if os.path.exists(file_path):  # File already exists
                    message = "File already uploaded. Change the name or try again."
                else:
                    uploaded_file.save(file_path)
                    message = f"Video uploaded to: {url_for('static', filename='videos/' + filename)}"
            else:
                message = "Unsupported file type"
    
    return render_template('upload.html', message=message)  # Pass message

if __name__ == '__main__':
    app.run(debug=True, port=5050)

