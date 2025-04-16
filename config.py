import os
import hmac
import hashlib
import time
from dotenv import load_dotenv
from flask import request, jsonify
from functools import wraps

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

    class RateLimiter:
        """In-memory rate limiting implementation"""
        def __init__(self):
            self.requests = {}

        def limit(self, key, limit, period):
            """Sliding window rate limiter decorator"""
            def decorator(f):
                @wraps(f)
                def wrapper(*args, **kwargs):
                    current = int(time.time())
                    window_key = f"{key}_{current // period}"
                    
                    if window_key not in self.requests:
                        self.requests[window_key] = []
                    
                    timestamps = self.requests[window_key]
                    timestamps = [t for t in timestamps if t > current - period]
                    
                    if len(timestamps) >= limit:
                        return jsonify({'error': 'Rate limit exceeded'}), 429
                    
                    timestamps.append(current)
                    self.requests[window_key] = timestamps
                    return f(*args, **kwargs)
                return wrapper
            return decorator

    class Authenticator:
        """Authentication handlers"""
        @staticmethod
        def hmac_auth(f):
            """HMAC signature validation decorator"""
            @wraps(f)
            def decorated(*args, **kwargs):
                received_sig = request.headers.get('X-Signature')
                computed_sig = hmac.new(
                    Config.Security.SECRET_KEY.encode(),
                    request.get_data(),
                    hashlib.sha256
                ).hexdigest()
                
                if not hmac.compare_digest(received_sig, computed_sig):
                    return jsonify({'error': 'Invalid HMAC signature'}), 401
                return f(*args, **kwargs)
            return decorated

        @staticmethod
        def api_key_auth(f):
            """API key validation decorator"""
            @wraps(f)
            def decorated(*args, **kwargs):
                api_key = request.headers.get('X-API-Key')
                if api_key not in Config.Security.API_KEYS:
                    return jsonify({'error': 'Invalid API key'}), 401
                return f(*args, **kwargs)
            return decorated

