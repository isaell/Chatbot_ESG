# ESG Annual Report Analysis System

Goal of the project: To create a specialized system for analyzing ESG (Environmental, Social, and Governance) content in annual reports using advanced NLP techniques and vector search.

⚠️ **_NOTE:_** This is work in progress with the basic functionality in place ⚠️

## Key Features

- **Automated Document Processing**: Downloads and processes annual reports from Google Drive
- **Advanced Text Analysis**: Uses an ESG-specialized model for better understanding of ESG content
- **Flexible Storage**: Supports both MongoDB and Qdrant vector databases
- **Interactive Query Interface**: Simple CLI for asking ESG-related questions

## Quick Start

1. **Setup Environment**

```bash
# Clone repository
git clone [repository-url]
cd [repository]

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

2. **Configure Credentials**

- Create `.env` file with database credentials:
```env
MONGODB_URI=your_mongodb_uri
QDRANT_HOST=your_qdrant_host
QDRANT_KEY=your_api_key
```
- Place Google Drive `credentials.json` in project root directory

3. **Run the System**

```bash
python pipeline.py
```

## Configuration

### Database Selection
Edit `pipeline.py` to choose your database:
```python
database_type = ('MongoDB' | 'Qdrant') # select database
```

### Chunking Parameters
Default settings in `ingestion.py`:
- Chunk size: 1024 tokens
- Overlap: 128 tokens
- Batch size: 32

## 💡 Usage

1. **Start the System**
```bash
python pipeline.py
```

2. **Query Interface**
- Enter ESG-related questions at the prompt
- Type 'quit' to exit
- Results show relevant text passages with source and confidence scores

## Architecture

### Components
- **Document Processor**: Handles PDF parsing and text extraction
- **Embedding Generator**: Creates ESG-specialized embeddings
- **Vector Storage**: Manages similarity search
- **Query Interface**: Handles user interactions

### Workflow
1. PDF documents downloaded from Google Drive
2. Text extracted and processed into chunks
3. Chunks embedded using finetuned ESG model
4. Embeddings stored in vector database
5. Query interface

## Development

### Adding Features
1. Create feature branch
2. Implement changes
3. Create an update tests
4. Submit pull request

<!-- ### Testing
```bash
python -m pytest tests/
``` -->

## Troubleshooting

### Common Issues
- **Database Connection**: Check credentials in `.env`
- **Google Drive Access**: Verify `credentials.json`
- **Memory Issues**: Adjust batch sizes in configuration

## Acknowledgments
Fine-tuned ESG model by Nithin Reddy: https://huggingface.co/nithinreddyy/finetuned-esg