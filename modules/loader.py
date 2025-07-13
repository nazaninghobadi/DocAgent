from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)

def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = UnstructuredPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    documents = loader.load()
    return documents
