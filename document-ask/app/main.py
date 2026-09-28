from dotenv import load_dotenv

load_dotenv()

from ingestion.loader import dir_loader
from ingestion.splitter import text_splitter


docs = dir_loader.load()

final_doc_string = " ".join(doc.page_content for doc in docs)

res = text_splitter.split_text(final_doc_string)
 



