import os

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc


class ContractProcessingService:
    def __init__(self, collection_name="contracts_business_latest_6"):
        load_dotenv()

        self.collection_name = collection_name
        self.doc_converter = DocumentConverter(allowed_formats=[InputFormat.PDF])

        # Get credentials from environment
        self.weaviate_url = os.getenv("WEAVIATE_URL")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        self.client = None
        self.collection = None

    def connect(self):
        """Connect to Weaviate and initialize the collection"""
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.weaviate_url,
            auth_credentials=Auth.api_key(self.weaviate_api_key),
            headers={"X-OpenAI-Api-key": self.openai_api_key},
        )

        if not self.client.is_ready():
            raise ConnectionError("Failed to connect to Weaviate")

        # Create or get collection
        if not self.client.collections.exists(self.collection_name):
            self.collection = self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=wvc.config.Configure.Vectorizer().text2vec_openai(),
                generative_config=wvc.config.Configure.Generative.openai(),
            )
        else:
            self.collection = self.client.collections.get(self.collection_name)

        return self

    def process_documents(self, folder_path="knowledge/contracts/"):
        """Process all PDF documents in the specified folder and store chunks in Weaviate"""
        if not self.client or not self.collection:
            self.connect()

        contracts_objs = list()

        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, filename)
                print(f"Processing {pdf_path}")

                result = self.doc_converter.convert(pdf_path)

                # Chunk the converted document
                for chunk in HybridChunker().chunk(result.document):
                    chunk_dict = {
                        "text": chunk.text,
                        "properties": {
                            "source_file": filename,
                            "heading": chunk.meta.headings[0]
                            if chunk.meta.headings
                            else "",
                            "page_number": chunk.meta.doc_items[0].prov[0].page_no
                            if chunk.meta.doc_items
                            else None,
                        },
                    }
                    contracts_objs.append(chunk_dict)

        # Insert data if we have any objects
        if contracts_objs:
            self.collection.data.insert_many(contracts_objs)
            return len(contracts_objs)
        return 0

    def close(self):
        """Close the Weaviate client connection"""
        if self.client:
            self.client.close()

    def __enter__(self):
        """Support for context manager pattern"""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection when exiting context"""
        self.close()


# Example usage:
# service = ContractProcessingService()
# with service:
#     service.process_documents()
