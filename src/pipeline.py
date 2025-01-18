import logging
from utils import setup_logging
from ingestion import get_gdrive_data, process_reports
from retrieval import RetrievalService

def main():
    setup_logging(log_level='INFO')  #WARNING
    logger = logging.getLogger(__name__)

    ##### Config
    # folder ID is for the folder where data is stored on google drive, here: 'annual_reports_2024'
    folder_id='1ud75JBvZ1JDnhkMqGy7GVOSw1mva0_ev'

    ## Database selector
    database_type = 'Qdrant'  # MongoDB or Qdrant

    # # OBS: Depending on which database you want to use either 1 or 2 params are needed
    if database_type == 'MongoDB':
        db_params = {'db_name': 'annual_reports', 'collection_name': 'document_embeddings'}
    elif database_type == 'Qdrant':
        db_params = {'collection_name': 'ESG_embeddings'}
    else:
        raise ValueError(f"Invalid database type: {database_type}")


    try:
        ### STEP 1: Data collection: Assemble input data for chatbot
        # here: pdfs are being taken from google drive
        # logger.info("Fetching documents from Google Drive")
        # get_gdrive_data(folder_id)

        # ### STEP 2: Processing & Storage - Ingestion pipeline
        # # processing step for parsing pdfs, chunking and creating embeddings in a vector database of your choice
        # logger.info("Processing reports")
        # process_reports(**db_params)

        ### STEP 3: Retrieval setup - Querying pipeline
        logger.info("Setting up retrieval")
        with RetrievalService(database_type, **db_params) as retrieval_service:
            retrieval_service.handle_user_query()

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()