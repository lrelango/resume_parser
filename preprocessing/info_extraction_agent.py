from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os
import json

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

def extract_info_agent(document_text):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    doc_tool = Tool(
        name="Document Loader",
        func=lambda x: x,
        description="Loads the resume document text for extraction."
    )
    format_instructions = output_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
    example_output = (
        "{{\n"
        '  "name": "Jane Doe",\n'
        '  "age": "",\n'
        '  "gender": "",\n'
        '  "location": "Mumbai",\n'
        '  "email": "jane.doe@email.com",\n'
        '  "phone": "+91-1234567890",\n'
        '  "qualification": "B.Tech Computer Science",\n'
        '  "experience": "2 Years",\n'
        '  "skills": ["Python", "SQL", "HTML"],\n'
        '  "candidate_summary": "Enthusiastic developer with 2 years of experience."\n'
        "}}"
    )
    prompt = (
        "You are an expert resume parser. Extract the following information from the resume text below. "
        "If a field is not present, leave it blank. Return ONLY the result as JSON in the format below. "
        "Do not include any extra text.\n"
        f"{format_instructions}\n"
        "Example Resume Text:\n"
        "Name: Jane Doe\nEmail: jane.doe@email.com\nPhone: +91-1234567890\nLocation: Mumbai\n"
        "Qualification: B.Tech Computer Science\nExperience: 2 Years\nSkills: Python, SQL, HTML\n"
        "Candidate Summary: Enthusiastic developer with 2 years of experience.\n"
        "Example Output:\n"
        f"{example_output}\n"
        "Resume Text:\n"
        "{input}"
    )
    agent = initialize_agent(
        tools=[doc_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    result = agent.invoke({"input": prompt.format(input=document_text)})
    print("\n[DEBUG] Raw LLM output:\n", result)
    output_str = result.get("output", "")
    output_str = output_str.strip()
    if output_str.startswith("```json"):
        output_str = output_str[len("```json"):].strip()
    if output_str.endswith("```"):
        output_str = output_str[:-3].strip()
    try:
        return json.loads(output_str)
    except Exception as e:
        print("[ERROR] Could not parse cleaned output as JSON:", e)
        return output_str