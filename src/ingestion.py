import os
import json
import io
import logging
import torch
import fitz
from typing import List, Dict, Any
from langchain.text_splitter import MarkdownTextSplitter
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
from sentence_transformers import SentenceTransformer
# https://github.com/huggingface/transformers/issues/5486
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from google.oauth2.credentials import Credentials       # To collect credentials
from google.auth.transport.requests import Request      # For refreshing credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # For handling new OAuth 2.0 authentiction
from googleapiclient.discovery import build             # To interact with Google APIs
from googleapiclient.http import MediaIoBaseDownload    # To download data from google drive

import utils
from Qdrant_handler import QdrantHandler
from MongoDB_handler import MongoDBHandler


class GoogleDriveService:
    def __init__(self):
        """
        Initialize Google Drive service with optional credentials path

        :drive.readonly: View and download all Drive files.
        :drive.metadata.readonly: View metadata for files in Drive.
        https://developers.google.com/drive/api/guides/api-specific-auth

        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first time.
        Currently, the token is created to authenticate both scopes.
        """
        self.logger = logging.getLogger(__name__)

        # If modifying these scopes, delete the file token.json
        SCOPES = [
            'https://www.googleapis.com/auth/drive.readonly',
            'https://www.googleapis.com/auth/drive.metadata.readonly'
        ]
        # Initialize credentials variable
        creds = None

        # Check if the token.json file exists, which stores the user's credentials after the initial login
        if os.path.exists('token.json'):
            try:
                with open('token.json', 'r') as token:
                    # Load credentials from the token file
                    creds = Credentials.from_authorized_user_info(json.load(token), SCOPES)

                # If the credentials are expired but a refresh token is available, try to refresh them
                if creds and creds.expired:
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        self.logger.info(f"Token refresh failed: {e}")
                        creds = None  # Reset creds to trigger new flow
                        if os.path.exists('token.json'):
                            os.remove('token.json')

            except Exception as e:
                self.logger.info(f"Error loading credentials: {e}")
                creds = None
                if os.path.exists('token.json'):
                    os.remove('token.json')

        # If there are no valid credentials (missing or expired),  start fresh OAuth flow to handle authentication
        if not creds or not creds.valid:
                # If there are no (valid) credentials available, initiate a new login using the OAuth 2.0 client secrets file
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                # Launch a local server to handle the OAuth redirect and login
                creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

        # Return Google Drive API service
        self.service = build('drive', 'v3', credentials=creds)
        self.logger.info('Google Drive service established')


    def get_metadata(self, query: str = None) -> list:
        """
        Collects metadata from given items returned by Google Drive API

        Args:
            query (Optional[str]): Search query for filtering files

        Returns:
            List of files matching the query
        """
        try:
            results = self.service.files().list(
                q = query,
                spaces = 'drive',
                fields = "files(id, name, mimeType, size, modifiedTime)"
            ).execute()
            return results.get('files', [])

        except Exception as e:
            raise RuntimeError(f"Error listing files: {e}")


    def download_files(self, file_id, file_name, download_dir: str = "downloads"):
        """
        Download files from Google Drive to a local directory.

        Args:
            Directory to download to [str], default: "downloads"
        """
        # Ensure the directory exists
        if not os.path.exists(download_dir):
            utils.ensure_dir(download_dir)

        # Path for the downloaded file
        file_path = os.path.join(download_dir, file_name)

        try:
            # API request to get and download the file
            request = self.service.files().get_media(fileId=file_id)
            fh = io.FileIO(file_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False

            # show progress
            while not done:
                status, done = downloader.next_chunk()
                self.logger.info(f"Downloaded {int(status.progress() * 100)}% of {file_name}.")

            # indicate where file is downloaded to
            self.logger.info(f"File downloaded to: {file_path}")

        except Exception as e:
            raise RuntimeError(f"Error downloading files: {e}")


class ParseChunkEmbed:
    def __init__(
            self,
            chunk_size: int = 1024,
            chunk_overlap: int = 128,
            model_name: str = "nithinreddyy/finetuned-esg", # general model alternative: 'sentence-transformers/all-mpnet-base-v2'
            batch_size: int = 32
            ):
        """
        Initialize with sentence-transformers model and processing parameters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        # Get logger for this class
        self.logger = logging.getLogger(__name__)

        # initialize sentence transformer
        self.logger.info(f'Loading {model_name} model...')
        self.model = SentenceTransformer(model_name)

        # Setup device for my M1 Pro
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.logger.info(f"Initializing model {model_name} on {self.device}")


    def parse_pdfs(self, pdf_path: str, pdf_directory: str) -> List[Dict]:
        """Parse PDF with enhanced table and image extraction"""
        try:
            # Use PyMuPDF (fitz) for basic text and structure
            doc = fitz.open(pdf_path)
            text_blocks = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Get basic text
                text = page.get_text()

                # Extract tables using built-in table detection
                tables = page.find_tables()
                tables_text = []
                for table in tables:
                    table_md = table.to_markdown()
                    tables_text.append(table_md)

                # Get image text using Tesseract OCR
                images = page.get_images(full=True)
                image_texts = []
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        # Convert CMYK to RGB if necessary
                        if pix.n - pix.alpha >= 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        # Save image temporarily
                        img_path = f"{pdf_directory}/temp_img_{page_num}_{img_index}.png"
                        pix.save(img_path)

                        # Use Tesseract OCR
                        try:
                            # Open image with PIL
                            image = Image.open(img_path)
                            # Extract text using Tesseract
                            img_text = pytesseract.image_to_string(image)
                            if img_text.strip():
                                image_texts.append(f"[Image Content: {img_text.strip()}]")
                        except Exception as ocr_error:
                            self.logger.warning(f"OCR failed for image on page {page_num + 1}: {ocr_error}")

                        # Clean up
                        os.remove(img_path)
                        pix = None

                    except Exception as e:
                        self.logger.warning(f"Failed to process image on page {page_num + 1}: {e}")

                # Combine all content
                combined_text = text
                if tables_text:
                    combined_text += "\n\n[Tables]\n" + "\n".join(tables_text)
                if image_texts:
                    combined_text += "\n\n[Images]\n" + "\n".join(image_texts)

                if combined_text.strip():
                    text_blocks.append({
                        'text': combined_text,
                        'page_number': page_num + 1,
                        'has_tables': bool(tables_text),
                        'has_images': bool(image_texts)
                    })

            doc.close()
            return text_blocks

        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {str(e)}", exc_info=True)
            return []


    def chunk_text(self, text_blocks: List[Dict]) -> List[Dict[str, Any]]:
        """Chunk text while preserving structure and metadata"""
        if not text_blocks:
            return []

        try:
            processed_chunks = []
            for block in text_blocks:
                # Split text into sections based on content type
                text_parts = block['text'].split('\n\n[Tables]')
                main_text = text_parts[0]
                tables_text = text_parts[1].split('\n\n[Images]')[0] if len(text_parts) > 1 else ""
                images_text = text_parts[1].split('\n\n[Images]')[1] if len(text_parts) > 1 and '[Images]' in text_parts[1] else ""

                # Process main text
                splitter = MarkdownTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )

                # Chunk main text
                main_chunks = splitter.create_documents([main_text])

                # Process each chunk
                for i, chunk in enumerate(main_chunks):
                    chunk_text = getattr(chunk, 'page_content', str(chunk))

                    # Add relevant table content if it exists
                    if tables_text and i == len(main_chunks) - 1:  # Add tables to last chunk of the page
                        chunk_text += f"\n\n[Tables]\n{tables_text}"

                    # Add relevant image content if it exists
                    if images_text and i == 0:  # Add images to first chunk of the page
                        chunk_text += f"\n\n[Images]\n{images_text}"

                    processed_chunks.append({
                        'text': chunk_text,
                        'chunk_id': len(processed_chunks) + i,
                        'metadata': {
                            'page_number': block['page_number'],
                            'has_tables': block['has_tables'],
                            'has_images': block['has_images']
                        }
                    })

            self.logger.debug(f"Created {len(processed_chunks)} chunks with preserved structure")
            return processed_chunks

        except Exception as e:
            self.logger.error(f"Error chunking text: {str(e)}", exc_info=True)
            return []


    @torch.no_grad()  # disables gradient calculation, reducing memory usage and speeds up computations
    def compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Creates embeddings using sentence-transformers model.

        Args:
            texts List[str]: List of text chunks to embed
        Returns:
            torch.Tensor: Tensor containing embeddings
        """
        self.logger.info(f"Computing embeddings for {len(texts)} texts")

        try:
            # Process in batches and show progress
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.device
            )

            # Move to CPU
            embeddings = embeddings.cpu()

            # Clear memory
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            return embeddings

        except Exception as e:
            self.logger.error(f"Error computing embeddings: {str(e)}", exc_info=True)
            return torch.tensor([])

def get_gdrive_data(folder_id):
    """
    Main fun that handles authentication for google drive, collects metadata and downloades local copy of files
    """
    #create an instance of the class GoogleDriveService
    gdrive_service = GoogleDriveService()

    # collect list with files
    files = gdrive_service.get_metadata(query = f"'{folder_id}' in parents")

    for file in files:
        # make sure the size component is readable
        human_readable_size = utils.get_size_format(int(file.get('size', 0)))

        # print metadata for each file
        print(f"Name: {file['name']}, ID: {file['id']}, Type: {file['mimeType']},"
            f"Size: {human_readable_size}, Modified: {file['modifiedTime']}")

        # downloads file if it's a pdf
        if file['name'].endswith('.pdf'):
            gdrive_service.download_files(file['id'], file['name'])



def process_reports(database, **kwargs):
    """
    Processes PDF reports and stores embeddings in specified database.

    Args:
        database (str): Database type (currently 'MongoDB' or 'Qdrant')
        **kwargs: Database-specific parameters
            - MongoDB: db_name, collection_name
            - Qdrant: collection_name
    """
    logger = logging.getLogger(__name__)

    # Database handlers with parameter validation
    handlers = {
        'MongoDB': (MongoDBHandler, ['db_name', 'collection_name']),
        'Qdrant': (QdrantHandler, ['collection_name'])
    }

    if database not in handlers:
        raise ValueError(f'Unsupported database type: {database}')

    # make sure all required keys are provided
    HandlerClass, required_keys = handlers[database]
    if not all(key in kwargs for key in required_keys):
        raise ValueError(f'{database} requires: {required_keys}')

    # Initialize components
    db_handler = HandlerClass(**{k: kwargs[k] for k in required_keys})
    ingester = ParseChunkEmbed()
    pdf_directory = 'downloads'

    if not os.path.exists(pdf_directory):
        logger.warning(f"PDF directory {pdf_directory} does not exist")
        return

    # Process each PDF file
    with db_handler as db:
        pdfs = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        logger.info(f"Found {len(pdfs)} PDFs to process")

        for filename in pdfs:
            pdf_path = os.path.join(pdf_directory, filename)
            logger.info(f"Processing: {filename}")

            try:
                # Parse PDF to markdown
                md_text = ingester.parse_pdfs(pdf_path, pdf_directory)
                if not md_text:
                    logger.warning(f"No text extracted from {filename}")
                    continue

                # Create chunks
                chunks = ingester.chunk_text(md_text)
                if not chunks:
                    logger.warning(f"No chunks created for {filename}")
                    continue

                # Get embeddings for all chunks
                texts = [chunk['text'] for chunk in chunks]
                embeddings = ingester.compute_embeddings(texts)

                if embeddings.numel() == 0:
                    logger.warning(f"No embeddings generated for {filename}")
                    continue

                # Store in database
                # currently implemented for batch processing
                db.insert_embeddings_batch(filename, chunks, embeddings)
                logger.info(f"Successfully processed {filename}")

            except Exception as e:
                logger.error(f"Failed processing {filename}: {str(e)}", exc_info=True)
                continue