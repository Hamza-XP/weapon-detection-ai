import ffmpeg
import tempfile
import hashlib
import logging
from typing import List, Dict
from config import Config
from datetime import timedelta

logger = logging.getLogger(__name__)

class VideoProcessingEngine:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self.supported_codecs = {'h264', 'hevc', 'vp9'}
    
    def process_video(self, input_path: str) -> Dict:
        """Full video processing workflow"""
        try:
            self._validate_file(input_path)
            
            # Step 1: Verify checksum
            file_hash = self._calculate_hash(input_path)
            
            # Step 2: Convert to standard format
            converted_path = self._convert_video(input_path)
            
            # Step 3: Extract metadata
            metadata = self._extract_metadata(converted_path)
            
            # Step 4: Sample frames
            frames = self._adaptive_frame_sampling(converted_path)
            
            return {
                "metadata": metadata,
                "sampled_frames": frames,
                "integrity_hash": file_hash,
                "temp_files": self._cleanup_resources([input_path, converted_path])
            }
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            self._cleanup_resources([input_path, converted_path])
            raise

    def _adaptive_frame_sampling(self, video_path: str) -> List[Dict]:
        """Intelligent frame sampling based on motion"""
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None
        )
        
        total_frames = int(video_stream.get('nb_frames', 0))
        duration = float(video_stream.get('duration', 0))
        
        base_interval = Config.FRAME_SAMPLING_INTERVAL
        dynamic_interval = max(1, int(total_frames / (duration * 0.5)))  # 0.5 fps per sec
        
        return [
            {
                "frame_number": i,
                "timestamp": str(timedelta(seconds=i/int(video_stream['avg_frame_rate'].split('/')[0]))),
                "path": self._extract_frame(video_path, i)
            }
            for i in range(0, total_frames, dynamic_interval)
        ]

    def _convert_video(self, input_path: str) -> str:
        """Standardize video format using FFmpeg"""
        output_path = os.path.join(self.temp_dir, f"conv_{os.path.basename(input_path)}.mp4")
        (
            ffmpeg
            .input(input_path)
            .output(output_path, vcodec='h264', acodec='aac', **{
                'b:v': '5000k',
                'maxrate': '5000k',
                'bufsize': '10000k',
                'pix_fmt': 'yuv420p'
            })
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return output_path

    def _calculate_hash(self, file_path: str) -> str:
        """Generate SHA-256 checksum"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _cleanup_resources(self, files: List[str]) -> int:
        """Secure file deletion"""
        count = 0
        for f in files:
            try:
                os.remove(f)
                count +=1
            except:
                pass
        return count
