# CRITICAL: Set environment variables BEFORE importing any libraries
# This prevents libraries from writing to /home/sbx_user1051 (read-only in Lambda)
# These can be overridden via Lambda Console -> Configuration -> Environment variables
import os

_tmp = os.environ.get("LAMBDA_TMP_DIR", "/tmp")
os.environ.setdefault("HOME", _tmp)
os.environ.setdefault("TMPDIR", _tmp)
os.environ.setdefault("TMP", _tmp)
os.environ.setdefault("TEMP", _tmp)
os.environ.setdefault("TORCH_HOME", f"{_tmp}/torch")
os.environ.setdefault("HF_HOME", f"{_tmp}/huggingface")
os.environ.setdefault("EASYOCR_MODULE_PATH", _tmp)
os.environ.setdefault("MODULE_PATH", _tmp)
os.environ.setdefault("XDG_CACHE_HOME", f"{_tmp}/.cache")
os.environ.setdefault("XDG_CONFIG_HOME", f"{_tmp}/.config")
os.environ.setdefault("XDG_DATA_HOME", f"{_tmp}/.local/share")

# Now safe to import other libraries
import base64
import logging
import string
import uuid
import zipfile
from io import BytesIO

from PIL import Image
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.pipeline_options import PipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DocumentFormatError(Exception):
    pass


class DocumentType:
    PDF = ".pdf"
    DOC = ".doc"
    DOCX = ".docx"
    RTF = ".rtf"
    IMAGE = ".png"
    TXT = ".txt"
    XLSX = ".xlsx"


class DocumentDetector:
    def __init__(self, bytes_stream: bytes):
        self.bytes_stream = bytes_stream

    def _detect_document_type(self) -> str:
        """Detect document type using file signatures"""
        magic_numbers = {
            b"%PDF": DocumentType.PDF,
            b"\xd0\xcf\x11\xe0": DocumentType.DOC,
            b"{\\\rtf1": DocumentType.RTF,
        }
        header = self.bytes_stream[:4]
        for signature, doc_type in magic_numbers.items():
            if header.startswith(signature):
                return doc_type
        if header.startswith(b"PK\x03\x04"):
            try:
                with zipfile.ZipFile(BytesIO(self.bytes_stream)) as zf:
                    if any(f.endswith("word/document.xml") for f in zf.namelist()):
                        return DocumentType.DOCX
                    if any(f.endswith("xl/workbook.xml") for f in zf.namelist()):
                        return DocumentType.XLSX
            except zipfile.BadZipFile:
                pass
        try:
            Image.open(BytesIO(self.bytes_stream)).verify()
            return DocumentType.IMAGE
        except Exception:
            pass
        if all(chr(b) in string.printable for b in self.bytes_stream[:100]):
            return DocumentType.TXT
        raise DocumentFormatError("Unsupported document format")


class DoclingParser(DocumentDetector):
    def __init__(
        self,
        base64_content: str = None,
        bytes_content: bytes = None,
        is_image_present: bool = False,
        is_md_response: bool = True,
        enable_table_extraction: bool = True,
        enable_ocr: bool = False,
        page_range: list = None,
    ):
        """Initialize parser with base64 encoded content or raw bytes

        Args:
            base64_content: Base64 encoded document content
            bytes_content: Raw bytes of the document
            is_image_present: Enable OCR for image-based PDFs (legacy param, use enable_ocr)
            is_md_response: Return markdown format (True) or plain text (False)
            enable_table_extraction: Extract tables with structure (default: True)
            enable_ocr: Enable OCR for scanned documents (default: False)
            page_range: List of page numbers to extract [start, end] (1-indexed, inclusive)
                       e.g., [1, 5] extracts pages 1-5, [3, 3] extracts only page 3
                       None = extract all pages
        """
        if base64_content is not None:
            self.bytes_stream = base64.standard_b64decode(base64_content)
        elif bytes_content is not None:
            self.bytes_stream = bytes_content
        else:
            raise ValueError("Either base64_content or bytes_content must be provided")

        if len(self.bytes_stream) < 10:
            raise ValueError("File must be provided")

        super().__init__(self.bytes_stream)
        self.doc_type = self._detect_document_type()
        self.is_image_present = is_image_present
        self.is_md_response = is_md_response
        self.enable_table_extraction = enable_table_extraction
        self.enable_ocr = enable_ocr or is_image_present  # Support legacy param
        self.page_range = page_range  # [start_page, end_page] (1-indexed)

    def _configure_converter(self):
        """Configure converter based on options"""
        if self.doc_type == ".pdf":
            pipeline_options = PdfPipelineOptions()

            # OCR settings
            pipeline_options.do_ocr = self.enable_ocr
            if self.enable_ocr:
                pipeline_options.force_full_page_ocr = True

            # Table extraction settings
            pipeline_options.do_table_structure = self.enable_table_extraction
            if self.enable_table_extraction:
                pipeline_options.table_structure_options.do_cell_matching = True

            accelerator_options = AcceleratorOptions()
            accelerator_options.device = AcceleratorDevice.CPU
            accelerator_options.num_threads = 8

            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=DoclingParseV2DocumentBackend,
                        accelerator_options=accelerator_options,
                    )
                }
            )
            return doc_converter

        elif self.doc_type == ".docx":
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.DOCX: WordFormatOption(
                        backend=MsWordDocumentBackend,
                    )
                }
            )
            return doc_converter
        else:
            # Default converter for other formats
            doc_converter = DocumentConverter()
            return doc_converter

    def _get_page_numbers(self) -> list:
        """Get list of page numbers to process based on page_range"""
        if not self.page_range:
            return None  # Process all pages

        start_page = self.page_range[0] if len(self.page_range) > 0 else 1
        end_page = self.page_range[1] if len(self.page_range) > 1 else start_page

        # Return list of page numbers (0-indexed for internal use)
        return list(range(start_page - 1, end_page))

    def _get_document_source(self) -> DocumentStream:
        """Create document stream with appropriate extension"""
        buf = BytesIO(self.bytes_stream)
        return DocumentStream(name=f"{str(uuid.uuid4())}{self.doc_type}", stream=buf)

    def parse_documents(self) -> str:
        """Parse document to markdown with optimized settings"""
        try:
            source = self._get_document_source()
            converter = self._configure_converter()
            conversion_result = converter.convert(source)
            
            # Get page range info for logging
            page_info = ""
            if self.page_range:
                start_page = self.page_range[0] if len(self.page_range) > 0 else 1
                end_page = self.page_range[1] if len(self.page_range) > 1 else start_page
                page_info = f" (filtering pages {start_page}-{end_page})"
                logger.info(f"Processing with page filter: {start_page} to {end_page}")
            
            results = ""
            if self.is_md_response:
                full_md = conversion_result.document.export_to_markdown()
                if self.page_range:
                    results = self._filter_pages_markdown(full_md)
                else:
                    results = full_md
            else:
                result = conversion_result.document.export_to_dict()
                if self.page_range:
                    results = self._filter_pages_text(result)
                else:
                    results = " ".join(
                        [result_obj.get("text", " ") for result_obj in result.get("texts", [])]
                    )
            
            return results
        except Exception as e:
            logger.error(f"Error parsing document: {str(e)}")
            raise DocumentFormatError(f"Failed to parse document: {str(e)}")
    
    def _filter_pages_markdown(self, markdown: str) -> str:
        """Filter markdown content by page range using page markers"""
        import re
        
        if not self.page_range:
            return markdown
        
        start_page = self.page_range[0] if len(self.page_range) > 0 else 1
        end_page = self.page_range[1] if len(self.page_range) > 1 else start_page
        
        # Split by common page markers
        # Docling uses "<!-- Page X -->" or similar patterns
        page_pattern = r'(<!-- Page \d+ -->|## Page \d+|\n---\n)'
        parts = re.split(page_pattern, markdown)
        
        # If no clear page markers, return full content
        if len(parts) <= 1:
            logger.info("No page markers found, returning full document")
            return markdown
        
        # Try to extract by page markers
        filtered_content = []
        current_page = 1
        
        for i, part in enumerate(parts):
            # Check if this is a page marker
            page_match = re.search(r'Page (\d+)', part, re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1))
                if start_page <= current_page <= end_page:
                    filtered_content.append(part)
            elif start_page <= current_page <= end_page:
                filtered_content.append(part)
        
        if filtered_content:
            return "".join(filtered_content).strip()
        
        # Fallback: return full content
        logger.info("Page filtering returned empty, returning full document")
        return markdown
    
    def _filter_pages_text(self, doc_dict: dict) -> str:
        """Filter text content by page range from document dict"""
        if not self.page_range:
            return " ".join([t.get("text", " ") for t in doc_dict.get("texts", [])])
        
        start_page = self.page_range[0] if len(self.page_range) > 0 else 1
        end_page = self.page_range[1] if len(self.page_range) > 1 else start_page
        
        filtered_texts = []
        for text_obj in doc_dict.get("texts", []):
            # Try to get page number from provenance
            prov = text_obj.get("prov", [])
            if prov and len(prov) > 0:
                page_no = prov[0].get("page_no", 1)
            else:
                page_no = 1
            
            if start_page <= page_no <= end_page:
                filtered_texts.append(text_obj.get("text", ""))
        
        return " ".join(filtered_texts) if filtered_texts else ""
