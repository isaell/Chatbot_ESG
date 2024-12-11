from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


documents = SimpleDirectoryReader('/Users/isabel.ellerbrock/Docker/chatbot_annual_reports/downloads').load_data()
#pdf_directory = '/Users/isabel.ellerbrock/Docker/chatbot_annual_reports/downloads'
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("How many employees does tescos have?")
print(response)