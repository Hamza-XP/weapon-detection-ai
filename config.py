import os
from dotenv import load_dotenv

load_dotenv()

class Config:
"""Central configuration class for the application"""
    
    # Security Configuration
    class Security:
        SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
        API_KEYS = os.getenv('API_KEYS', 'default_key').split(',')
        RATE_LIMIT = int(os.getenv('RATE_LIMIT', '100'))  # Requests per minute
        TOKEN_EXPIRY = int(os.getenv('TOKEN_EXPIRY', '3600'))  # Seconds
    
    # Application Configuration
    class App:
        MODEL_PATH = os.getenv('MODEL_PATH', 'models/exported_model')
        MAX_IMAGE_SIZE_MB = 10
        MAX_VIDEO_SIZE_MB = 1024
        FRAME_SAMPLING_INTERVAL = 20
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}
        MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '1073741824'))  # 1GB
        
        UPLOAD_FOLDERS = {
            'images': os.getenv('UPLOAD_IMAGE_DIR', 'uploads/images'),
            'videos': os.getenv('UPLOAD_VIDEO_DIR', 'uploads/videos')
        }
    }
