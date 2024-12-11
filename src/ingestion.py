#!/usr/bin/python3

import utils
import os
import json 
import io
import PyPDF2
import torch

from google.oauth2.credentials import Credentials       # To collect credentials
from google.auth.transport.requests import Request      # For refreshing credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # For handling new OAuth 2.0 authentiction
from googleapiclient.discovery import build             # To interact with Google APIs
from googleapiclient.http import MediaIoBaseDownload    # To download data from google drive


from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer #, AutoModelForSequenceClassification, pipeline


# Call the logging setup function
utils.setup_logging(log_level='WARNING')  #DEBUG


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
            with open('token.json', 'r') as token:
                # Load credentials from the token file
                creds = Credentials.from_authorized_user_info(json.load(token), SCOPES)
        
        # If there are no valid credentials (missing or expired), handle authentication
        if not creds or not creds.valid:
            # If the credentials are expired but a refresh token is available, refresh them
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
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


#
    def get_metadata(self, query: str = None) -> list:
        """
        Collects metadata from given items returned by Google Drive API

        Input: query (Optional[str]): Search query for filtering files
        >> Returns: List of files matching the query
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
        

# 
    def download_files(self, file_id, file_name, download_dir: str = "downloads"):
        """
        Download files from Google Drive to a local directory.
        
        Input: Directory to download to
        """
        # Ensure the download directory exists
        if not os.path.exists(download_dir):
            #os.makedirs(download_dir)
            utils.ensure_dir(download_dir)

        # Path for the downloaded file
        file_path = os.path.join(download_dir, file_name)
        
        try:
            # API request to get and download the file
            request = self.service.files().get_media(fileId=file_id)
            fh = io.FileIO(file_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            
            while not done:
                # print progress while file not completely downloaded
                status, done = downloader.next_chunk()
                print(f"Downloaded {int(status.progress() * 100)}% of {file_name}.")

            # indicate where file is downloaded to 
            print(f"File downloaded to: {file_path}")

        except Exception as e:
            raise RuntimeError(f"Error downloading files: {e}")



class ParseChunkEmbed:
    
    def __init__(self):

        # ESG-BERT model
        self.tokenizer = AutoTokenizer.from_pretrained('nbroad/ESG-BERT')
        self.model = AutoModel.from_pretrained('nbroad/ESG-BERT')


    def parse_pdfs(self, pdf_path):
        """
        Input: path for input pdfs to be parsed 
        >> returns: text
        """        
        with io.FileIO(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ' '.join([page.extract_text() for page in reader.pages])
        return text
   

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """
        chunks input text, currently in a recursive style

        Input: text 
        >> returns: chunked text
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size, 
            chunk_overlap = overlap
        )
        return splitter.split_text(text)


    def create_embedding(self, text):
        """
        creates embeddings using the pretrained tokenizer from ESG-BERT,
        transform text into vector representations that capture semantic meaning.

        Input: text 
        >> returns: embeddings
        """
        # output should be in PyTorch tensor format, max_length is a common limit for BERT models.
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        # torch.no_grad disables gradient calculation, reducing memory usage and speeds up computations 
        with torch.no_grad():
            # passes the tokenized inputs through the BERT model to obtain outputs
            outputs = self.model(**inputs)
            # Contains the hidden states of the last layer of the BERT model for each token in the input sequence
            # .mean(dim=1): Computes the mean across all tokens in the sequence (dim 1), resulting in a single vector 
            # representation for the entire input text. This approach is commonly used to create sentence-level embeddings.
	        # .squeeze(): Removes any singleton dimensions from the tensor
	        # .numpy(): Converts the PyTorch tensor into a NumPy array for easier manipulation and storage
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        # Returns the computed embedding as a NumPy array, representing the semantic meaning of the input text    
        return embedding



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
        print(f"Name: {file['name']}, ID: {file['id']}, Type: {file['mimeType']}, "
              f"Size: {human_readable_size}, Modified: {file['modifiedTime']}")

        # downloads file if it's a pdf
        if file['name'].endswith('.pdf'):
            gdrive_service.download_files(file['id'], file['name'])


def process_reports(db_name, collection_name):
    """
    Processes reports (parsing pdfs, chunking text, creating and inserting embeddings) 
    after having pdfs in a local directory. 

    For flexible use, this could be expanded to processing other input file types and
    getting data from other data sources than google drive.

    Input: database name and collection name on Mongo DB
    """
    ingester = ParseChunkEmbed()

    # Automatically handles connection and closure
    with utils.MongoDBHandler(db_name, collection_name) as db:
        pdf_directory = "downloads"  

        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)

                print(f'Parsing pdfs, chunking text and creating embeddings for: {filename}')
                
                # Parse and chunk PDF
                text = ingester.parse_pdfs(pdf_path)
                chunks = ingester.chunk_text(text)
                    
                # Create and store embeddings
                for chunk in chunks:
                    embedding = ingester.create_embedding(chunk)

                    # Imserts embeddings 
                    db.insert_embedding(filename, chunk, embedding)



# EXECUTE
# folder ID is for the folder where data is stored on google drive, here: 'annual_reports_2024'
get_gdrive_data(folder_id = '1ud75JBvZ1JDnhkMqGy7GVOSw1mva0_ev')
# insert database name and collection name (currently on mongo db) 
process_reports(db_name = 'esg_reports', collection_name = 'document_embeddings')
                                    