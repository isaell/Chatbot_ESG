
import os
import json
import io
import logging
import pymupdf4llm
import torch
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
from langchain.text_splitter import MarkdownTextSplitter # RecursiveCharacterTextSplitter

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
                        print(f"Token refresh failed: {e}")
                        creds = None  # Reset creds to trigger new flow
                        if os.path.exists('token.json'):
                            os.remove('token.json')

            except Exception as e:
                print(f"Error loading credentials: {e}")
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

        print('Google Drive service established')
        # Return Google Drive API service
        self.service = build('drive', 'v3', credentials=creds)


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
                print(f"Downloaded {int(status.progress() * 100)}% of {file_name}.")

            # indicate where file is downloaded to
            print(f"File downloaded to: {file_path}")

        except Exception as e:
            raise RuntimeError(f"Error downloading files: {e}")



class ParseChunkEmbed:
    def __init__(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
            model_name: str = 'nbroad/ESG-BERT',
            batch_size: int = 16
            ):
        """
        Initialize with ESG-BERT model and processing parameters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        # Get logger for this class
        self.logger = logging.getLogger(__name__)

        # initialize ESG-BERT model and tokenizer
        self.logger.info(f'Loading {model_name} model and tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Setup device for my M1 Pro
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.logger.info(f"Initializing model {model_name} on {self.device}")


    def parse_pdfs(self, pdf_path: str, pdf_directory: str) -> str:
        """
        Parse PDF to markdown format with image handling

        Args:
            pdf_path: Path to the PDF file
            pdf_directory: Directory to save extracted images

        Returns:
            str: Markdown formatted text
        """
        try:
            # Extract text in Markdown format
            md_text = pymupdf4llm.to_markdown(pdf_path, write_images=True, image_path=pdf_directory)

            if not md_text:
                self.logger.warning(f"No text extracted from {pdf_path}")
                return ""

            return md_text

        except Exception as e:
            print(f"Error parsing PDF {pdf_path}: {str(e)}", exc_info=True)
            return ""


    def chunk_text(self, md_text: str) -> List[Dict[str, str]]:
        """
        chunks input markdown text into smaller pieces while preserving markdown structure

        Args:
            md_text: Markdown formatted text

        Returns:
            List[Dict[str, str]]: List of document chunks with metadata
        """
        if not md_text:
            return []

        try:
            # Initialize the markdown splitter
            splitter = MarkdownTextSplitter(
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap
            )

            # Create documents from the markdown text
            chunks = splitter.create_documents([md_text])

            # Convert chunks to the format expected by the embedding model
            processed_chunks = [
                {
                    'text': getattr(chunk, 'page_content', str(chunk)),
                    'chunk_id': i,
                    'metadata': getattr(chunk, 'metadata', {})
                }
                for i, chunk in enumerate(chunks)
            ]

            self.logger.debug(f"Created {len(processed_chunks)} chunks")
            return processed_chunks

        except Exception as e:
            print(f"Error chunking text: {str(e)}", exc_info=True)
            return []


    @torch.no_grad()  # disables gradient calculation, reducing memory usage and speeds up computations
    def compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        creates embeddings using the pretrained tokenizer from ESG-BERT,
        transform text into vector representations that capture semantic meaning.
        Currently using memory optimization

        Args:
            text List[str]
        Returns:
            embeddings [torch tensor]
        """

        all_embeddings = []

        # Process in batches
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        #self.logger.info(f"Processing {len(texts)} texts in {n_batches} batches")
        self.logger.info(f"Computing embeddings for {len(texts)} texts")

        for i in tqdm(range(0, len(texts), self.batch_size)): # desc="Computing embeddings"
            batch_texts = texts[i:i + self.batch_size]

            try:
                # Tokenize the input text
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',  # should be in PyTorch tensor format
                    truncation=True,
                    padding=True,
                    max_length=512   # common limit for BERT models
                ).to(self.device)


                # passes the tokenized inputs through the BERT model to obtain outputs
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                # Contains the hidden states of the last layer of the BERT model for each token in the input sequence
                # .mean(dim=1): Computes the mean across all tokens in the sequence (dim 1), resulting in a single vector
                # representation for the entire input text. This approach is commonly used to create sentence-level embeddings.

                # Move to CPU and store
                all_embeddings.append(embeddings.cpu())

                # Clear memory
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                self.logger.debug(f"Processed batch {i//self.batch_size + 1}/{n_batches}")

            except Exception as e:
                self.logger.error(f"Error processing batch {i//self.batch_size + 1}: {str(e)}", exc_info=True)
                continue

        if not all_embeddings:
            self.logger.error("No embeddings were generated")
            return torch.tensor([])

        # Combine all embeddings
        return torch.cat(all_embeddings, dim=0)



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