import os
import uuid
import logging
from PIL import Image, ExifTags, UnidentifiedImageError
from io import BytesIO
from config import Config
from datetime import datetime

logger = logging.getLogger(_name_)

class ImageProcessor:
    def _init_(self):
        self.allowed_mime = {'image/jpeg', 'image/png', 'image/webp'}
        self.max_size = Config.MAX_IMAGE_SIZE_MB * 1024 * 1024

    async def process_upload(self, file_stream, filename):
        """Full image processing pipeline"""
        try:
            self._validate_filename(filename)
            image_data = await self._read_stream(file_stream)
            self._validate_size(image_data)
            clean_image = await self._sanitize_image(image_data)
            return await self._store_image(clean_image, filename)
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise

    def _validate_filename(self, filename):
        if not 3 < len(filename) < 120:
            raise ValueError("Invalid filename length")
        if any(c in filename for c in {'/', '\\', ':', '*', '?'}):
            raise ValueError("Invalid characters in filename")

    async def _read_stream(self, stream):
        """Read stream with timeout protection"""
        data = b''
        chunk_size = 4096
        start_time = datetime.now()
        while True:
            chunk = await stream.read(chunk_size)
            if not chunk:
                break
            if (datetime.now() - start_time).seconds > 30:
                raise TimeoutError("Upload timeout")
            data += chunk
            if len(data) > self.max_size:
                raise ValueError("File size exceeded")
        return data

    async def _sanitize_image(self, image_data):
        """Remove EXIF and convert formats"""
        try:
            with Image.open(BytesIO(image_data)) as img:
                # Strip EXIF
                data = list(img.getdata())
                clean_img = Image.new(img.mode, img.size)
                clean_img.putdata(data)
                
                # Convert to standardized format
                output = BytesIO()
                clean_img.save(output, format='WEBP', quality=85)
                return output.getvalue()
        except UnidentifiedImageError:
            raise ValueError("Invalid image file")

    async def _store_image(self, image_data, original_name):
        """Save with UUID and audit trail"""
        file_id = str(uuid.uuid4())
        ext = original_name.split('.')[-1].lower() if '.' in original_name else 'webp'
        filename = f"{file_id}.{ext}"
        path = os.path.join(Config.UPLOAD_FOLDERS['images'], filename)
        
        with open(path, 'wb') as f:
            f.write(image_data)
        
        return {
            "file_id": file_id,
            "original_name": original_name,
            "bytes": len(image_data),
            "sha256": await self._generate_checksum(image_data)
        }

    async def _generate_checksum(self, data):
        import hashlib
        return hashlib.sha256(data).hexdigest()
