import os
import cv2
import base64
from flask import Flask, render_template, request, jsonify
import numpy as np
from werkzeug.utils import secure_filename
from weapon_detector import WeaponDetector  # Assuming your class is in weapon_detector.py

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Initialize detector once when app starts
detector = WeaponDetector()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image
            try:
                image = cv2.imread(filepath)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = detector.detect(image_rgb)
                
                # Convert annotated image to base64 for display
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(results['annotated_image'], np.uint8), 
                    cv2.COLOR_BGR2RGB))
                annotated_img = base64.b64encode(buffer).decode('utf-8')
                
                return render_template('result.html',
                                       annotated_image=annotated_img,
                                       detections=zip(results['boxes'], results['scores'], results['classes']))
            except Exception as e:
                return render_template('index.html', error=f"Error processing image: {str(e)}")
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
        
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Read image directly from memory
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = detector.detect(image_rgb)
        
        # Convert annotated image to bytes
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(
            cv2.imdecode(np.frombuffer(results['annotated_image'], np.uint8), 
            cv2.COLOR_BGR2RGB))
        annotated_img = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'detections': {
                'count': len(results['boxes']),
                'boxes': results['boxes'],
                'scores': results['scores'],
                'classes': results['classes']
            },
            'annotated_image': annotated_img
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if _name_ == '_main_':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)
