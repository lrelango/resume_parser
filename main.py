from ingestion.file_loader import load_resume
from preprocessing.info_extraction_agent import extract_info_agent

def main():
    # Example: Load a sample resume (PDF or DOCX)
    file_path = "data/Resume_2.pdf"
    docs = load_resume(file_path)

    # Combine all chunks into one string
    full_text = " ".join([doc.page_content for doc in docs])

    print(f"Loaded {len(docs)} documents")
    print(f"\n--- Full Resume ---\n")
    #print(full_text[:3000])  # show first 3000 chars

    # Extract info using AI Agent
    extracted_info = extract_info_agent(full_text)
    print("\nExtracted Information (JSON):")
    print(extracted_info)

if __name__ == "__main__":
    main()