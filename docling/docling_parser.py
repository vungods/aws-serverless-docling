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
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from pydantic import BaseModel

logger = logging.getLogger(__name__)

import os
# Set all temporary directories to /tmp (Lambda only writable directory)
# These are set here as fallback, but should also be set in lambda_function.py
if "HOME" not in os.environ:
    os.environ["HOME"] = "/tmp"
os.environ["TMPDIR"] = "/tmp"
os.environ["TMP"] = "/tmp"
os.environ["TEMP"] = "/tmp"
os.environ["TORCH_HOME"] = "/tmp/torch"
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["EASYOCR_MODULE_PATH"] = "/tmp"
os.environ["MODULE_PATH"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
os.environ["XDG_CONFIG_HOME"] = "/tmp/.config"
os.environ["XDG_DATA_HOME"] = "/tmp/.local/share"



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
            b'%PDF': DocumentType.PDF,
            b'\xd0\xcf\x11\xe0': DocumentType.DOC,
            b'{\\\rtf1': DocumentType.RTF,
        }
        header = self.bytes_stream[:4]
        for signature, doc_type in magic_numbers.items():
            if header.startswith(signature):
                return doc_type
        if header.startswith(b'PK\x03\x04'):
            try:
                with zipfile.ZipFile(BytesIO(self.bytes_stream)) as zf:
                    if any(f.endswith('word/document.xml') for f in zf.namelist()):
                        return DocumentType.DOCX
                    if any(f.endswith('xl/workbook.xml') for f in zf.namelist()):
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
    def __init__(self, base64_content: str = None, bytes_content: bytes = None, is_image_present: bool = False,
                 is_md_response: bool = False):
        """Initialize parser with base64 encoded content or raw bytes"""
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


    def _configure_converter(self):
        """Configure optimized converter for speed"""
        match self.is_image_present:
            case False:
                if self.doc_type==".pdf":
                    pipeline_options = PdfPipelineOptions()
                    pipeline_options.do_ocr = False
                    pipeline_options.do_table_structure = False
                    pipeline_options.table_structure_options.do_cell_matching = False
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
                elif self.doc_type==".docx":
                    doc_converter = DocumentConverter(
                        format_options={
                            InputFormat.DOCX: WordFormatOption(
                                backend=MsWordDocumentBackend,

                                    )
                                }
                            )
                    return doc_converter
                else:
                    doc_converter = DocumentConverter()
                    return doc_converter

            case True:
                if self.doc_type==".pdf":
                    pipeline_options = PdfPipelineOptions(do_ocr=True,
                                                          force_full_page_ocr=True,
                                                          )
                    accelerator_options = AcceleratorOptions(num_threads=8,
                                                             device=AcceleratorDevice.CPU)
                    doc_converter = DocumentConverter(
                        format_options={
                            InputFormat.PDF: PdfFormatOption(
                                pipeline_options=pipeline_options,
                                accelerator_options=accelerator_options
                            )
                        }
                    )
                    return doc_converter
                else:
                    doc_converter = DocumentConverter( )
                    return doc_converter

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
            results = ""
            match self.is_md_response:
                case True:
                    results =conversion_result.document.export_to_markdown()
                case False:
                    result =conversion_result.document.export_to_dict()
                    results = " ".join([result_obj.get("text"," ") for result_obj in result.get('texts',[])])
            return results
        except Exception as e:
            logger.error(f"Error parsing document: {str(e)}")
            raise DocumentFormatError(f"Failed to parse document: {str(e)}")