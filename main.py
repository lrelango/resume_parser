import pandas as pd
from ingestion.file_loader import load_resume
from preprocessing.info_extraction_agent import extract_info_agent

def main():
    # Example: Load a sample resume (PDF or DOCX)
    file_paths = [
        "data/Resume_2.pdf",
        "data/resume_1.docx",
        "data/resume_2.docx"
    ]
    results=[]
    for file_path in file_paths:
        docs = load_resume(file_path)
        full_text = " ".join([doc.page_content for doc in docs])
        print(f"Loaded {len(docs)} documents from {file_path}")
        print(f"\n--- Full Resume ---\n")
        extracted_info = extract_info_agent(full_text)
        print("\nExtracted Information (JSON):")
        print(extracted_info)
        results.append(extracted_info)

    dict_results = [r for r in results if isinstance(r, dict)]

    df = pd.DataFrame(dict_results)
    print("\nExtracted DataFrame:")
    print(df)

    df.to_csv("extracted_resume_info.csv", index=False)

if __name__ == "__main__":
    main()