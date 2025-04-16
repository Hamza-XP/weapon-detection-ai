from datetime import datetime
from flask import make_response, jsonify
from typing import List, Dict, Union
from config import Config

class APIResponseBuilder:
    """Lightweight API response handler with enterprise features"""
    
    def build_response(self, 
                     data: Union[List, Dict], 
                     metadata: Dict = None,
                     status: int = 200) -> make_response:
        """Construct standardized API response"""
        response = {
            "api_version": "2.1",
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
            "metadata": metadata or {}
        }
        return make_response(jsonify(response), status

    def error_response(self, 
                      error_code: str, 
                      message: str, 
                      status: int = 400,
                      details: Dict = None) -> make_response:
        """Generate error responses with documentation links"""
        error = {
            "code": error_code,
            "message": message,
            "documentation": f"{Config.API_BASE_URL}/docs/errors/{error_code}",
            "details": details or {}
        }
        return make_response(jsonify({"error": error}), status

    def paginate(self, 
                data: List, 
                page: int, 
                per_page: int) -> make_response:
        """Paginate results without external dependencies"""
        total = len(data)
        return self.build_response(
            data=data[(page-1)*per_page : page*per_page],
            metadata={
                "pagination": {
                    "current_page": page,
                    "per_page": per_page,
                    "total_items": total,
                    "total_pages": (total + per_page - 1) // per_page
                }
            }
        )

class DetectionResultFormatter:
    """Formats model outputs for API consumption"""
    
    @staticmethod
    def format_detection(detection: Dict, image_size: tuple) -> Dict:
        """Convert raw detection to API-safe format"""
        width, height = image_size
        return {
            "label": detection["label"],
            "confidence": round(detection["confidence"], 4),
            "bounding_box": {
                "x_min": int(detection["bbox"][0] * width),
                "y_min": int(detection["bbox"][1] * height),
                "x_max": int(detection["bbox"][2] * width),
                "y_max": int(detection["bbox"][3] * height)
            },
            "timestamp": detection.get("timestamp")
        }

# Example usage:
# builder = APIResponseBuilder()
# formatted = [DetectionResultFormatter.format_detection(d, (640,480)) for d in raw_dets]
# response = builder.paginate(formatted, page=1, per_page=10)
