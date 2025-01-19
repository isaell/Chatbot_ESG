import logging
from utils import setup_logging
from ingestion import get_gdrive_data, process_reports
from retrieval import RetrievalService

class Pipeline:
    def __init__(self, config):
        self.config = config

        # set up logger
        setup_logging(log_level='INFO')
        self.logger = logging.getLogger(__name__)

        # database configuration
        self.db_type = config['database']['type']
        self.db_params = config['database']['params'][self.db_type]

    def collect_data(self):
        """Step 1: Data collection from Google Drive"""
        self.logger.info("Fetching documents from Google Drive")
        get_gdrive_data(self.config['folder_id'])

    def process_data(self):
        """Step 2: Processing and storing data"""
        self.logger.info("Processing reports")
        process_reports(self.db_type, **self.db_params)

    def setup_retrieval(self):
        """Step 3: Setting up and running retrieval"""
        self.logger.info("Setting up retrieval")
        with RetrievalService(self.db_type, **self.db_params) as retrieval_service:
            retrieval_service.handle_user_query()

    def run(self):
        """Main execution method, comment here to exclude steps"""
        try:
            self.collect_data()
            self.process_data()
            self.setup_retrieval()
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

def main():
    # Configuration
    config = {
        'folder_id': '1ud75JBvZ1JDnhkMqGy7GVOSw1mva0_ev',
        'database': {
            'type': 'Qdrant',  # or MongoDB, Qdrant
            'params': {
                'MongoDB': {'db_name': 'annual_reports', 'collection_name': 'document_embeddings'},
                'Qdrant': {'collection_name': 'ESG_embeddings'}
            }
        }
    }

    pipeline = Pipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()