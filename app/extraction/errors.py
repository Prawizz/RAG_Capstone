"""Error ระดับไฟล์ที่ extractor/detector โยนขึ้นมา pipeline จะแปลงเป็น ``Document.error``"""

from app.models import ErrorCode


class ExtractionError(Exception):
    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
