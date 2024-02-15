import os
from pathlib import Path
from llama_index.storage import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.readers import PDFReader


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        print("type of data", type(data)   )
        index = VectorStoreIndex.from_documents(data)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


pdf_path = os.path.join("data", "Canada.pdf")
print(pdf_path)
canada_pdf = PDFReader().load_data(file=Path(pdf_path))

canada_index = get_index(canada_pdf, "canada")
canada_engine = canada_index.as_query_engine()