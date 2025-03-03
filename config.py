import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Image processing
    MAX_IMAGE_SIZE_MB = 10
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Video processing
    MAX_VIDEO_SIZE_MB = 1024
    FRAME_SAMPLING_INTERVAL = 20
    
    # Model
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/exported_model')
    
    # Security
    API_KEYS = os.getenv('API_KEYS', 'default_key').split(',')
    
    # Storage
    UPLOAD_FOLDERS = {
        'images': 'uploads/images',
        'videos': 'uploads/videos'
    }
