from typing import List
import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    # Set up the Logfire span tracking
    with logfire.span("Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        # FIXED: Indented all processing logic inside the Logfire span block
        # Use LangChain to safely handle messy PPT/HTML strings
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]   # order of preference for splitting
        )

        valid_chunks = splitter.split_text(text)

        # Log the output count cleanly nested inside your active dashboard span
        logfire.info("Generated chunks count", count=len(valid_chunks))
        return valid_chunks

