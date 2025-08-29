from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the schema for extraction
response_schemas = [
    ResponseSchema(name="name", description="Full name of the candidate"),
    ResponseSchema(name="age", description="Age of the candidate"),
    ResponseSchema(name="gender", description="Gender of the candidate"),
    ResponseSchema(name="location", description="Location of the candidate"),
    ResponseSchema(name="email", description="Email address"),
    ResponseSchema(name="phone", description="Phone number"),
    ResponseSchema(name="qualification", description="Highest qualification"),
    ResponseSchema(name="experience", description="Total years of experience"),
    ResponseSchema(name="skills", description="List of skills"),
    ResponseSchema(name="candidate_summary", description="Brief summary of the candidate"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

def extract_info_agent(document_text: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

    format_instructions = output_parser.get_format_instructions()
    current_date = datetime.now().strftime("%B %d, %Y")

    prompt_template = PromptTemplate(
        template=(
            f"You are an expert resume parser. Today's date is {current_date}. "
            "Use this date to calculate age from date of birth or year of birth.\n"
            "Extract the following information from the resume text below. "
            "Return ONLY the result as JSON in the specified format. Do not include any extra text.\n"
            "Check for address in the text to extract location.\n\n"

            "For 'age': If date of birth (DOB) or year of birth is explicitly present in ANY common format "
            "(YYYY-MM-DD, DD-MM-YYYY, DD-MMM-YYYY, etc.), calculate the age using today's date. "
            "If not present, strictly return 'No Info'.\n\n"

            "For EVERY field (name, age, gender, location, email, phone, qualification, "
            "experience, skills, candidate_summary): If information is not explicitly present "
            "or cannot be determined, strictly return 'No Info'. Never leave a field blank.\n\n"

            "{format_instructions}\n\n"
            "Resume Text:\n{input}"
        ),
        input_variables=["input"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt_template | llm | output_parser

    try:
        with get_openai_callback() as cb:
            result = chain.invoke({"input": document_text})

            print("----- Token Usage & Cost -----")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost:.6f}")
            print("------------------------------")

        return result
    except Exception as e:
        print("[ERROR] Could not parse JSON:", e)
        return {"error": str(e)}

