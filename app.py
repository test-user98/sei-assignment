from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import google.generativeai as genai
import requests
import re
import json
from datetime import datetime
import os
from openai import OpenAI
from urllib.parse import urlparse


app = FastAPI()
API_TOKEN = os.getenv('API_TOKEN')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel("gemini-1.5-flash")

# Define directories
POLICY_DIR = "policy"
os.makedirs(POLICY_DIR, exist_ok=True)

# Input schemas
class TrainRequest(BaseModel):
    policy_name: str
    url: str

class ComplianceRequest(BaseModel):
    url: str

# Forbidden terms regex pattern list
FORBIDDEN_TERMS = []
def get_prompt(mode, raw_content, policy=None):
    if mode == 'training':
        with open("training_prompt.json", "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data['training_prompt'].format(raw_content=raw_content)

    elif mode == 'testing':
        with open("testing_prompt.json", "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data['testing_prompt'].format(raw_content=raw_content, policy_content=policy['content'])

    else:
        raise ValueError("Invalid mode, must be 'training' or 'testing'")
    


def fetch_and_process_content(url: str):
    try:
        # Fetch webpage content
        response = requests.get(url)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        main_tags = ['article', 'main']
        fallback_tags = ['div', 'span']
        content_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul']

        content = {"headers": [], "paragraphs": [], "lists": []}

        main_content = soup.find(main_tags) or soup.body

        def add_to_content(key, text):
            if text and text not in content[key]:
                content[key].append(text)

        for tag in main_content.find_all(content_tags, recursive=True):
            text = tag.get_text(strip=True)
            if tag.name.startswith('h'):
                add_to_content('headers', text)
            elif tag.name == 'p':
                add_to_content('paragraphs', text)
            elif tag.name == 'ul':
                items = [li.get_text(strip=True) for li in tag.find_all('li') if li.get_text(strip=True)]
                if items and items not in content['lists']:
                    content['lists'].append(items)

        for tag in main_content.find_all(fallback_tags, recursive=True):
            text = tag.get_text(strip=True)
            if len(text.split()) > 10:
                add_to_content('paragraphs', text)

        return " ".join(content['headers'] + content['paragraphs'] + [" ".join(lst) for lst in content['lists']])
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch webpage: {e}")

def llm_parsing(mode, raw_content, policy=None):
    prompt = get_prompt(mode, raw_content, policy)

    filename = "processed_llm_data.json"
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                # Use a more robust JSON parsing method
                file_content = file.read()
                structured_content = json.loads(sanitize_content(file_content))
            return structured_content
        except json.JSONDecodeError:
            print(f"Error reading {filename}. Regenerating file.")

    chat = model.start_chat(history=[
        {"role": "model", "parts": "You are an expert in extracting key details from webpages."},
        {"role": "user", "parts": prompt}
    ])

    response = chat.send_message(prompt)

    structured_content = { 
        "status": "success",
        "content": response.text, 
        "metadata": {
            "timestamp": str(datetime.now()),  
            "model": "Gemini"
        }
    }

    # Ensure safe writing
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(structured_content, file, ensure_ascii=False, indent=4)

    return structured_content

def sanitize_content(content: str) -> str:
    if not isinstance(content, str):
        content = str(content)
    return re.sub(r'[\x00-\x1F\x7F]', '', content)

def regex_check(content: str):
    findings = []
    for term in FORBIDDEN_TERMS:
        matches = re.finditer(term, content, re.IGNORECASE)
        for match in matches:
            findings.append({
                "term": match.group(),
                "context": content[max(0, match.start() - 50):match.end() + 50]
            })
    return findings


@app.post("/train")
def train_policy(request: TrainRequest, authorization: str = Header(...)):
    if authorization != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    
    llm_data_path = "processed_llm_data.json"
    if os.path.exists(llm_data_path):
        return {"message": "LLM data already processed and saved. Skipping API call to save costs."}

    content = fetch_and_process_content(request.url)
    llm_parsing('training', content)  # Removed policy argument

    # Save the policy content to a file
    policy_name = request.policy_name if request.policy_name else "default_policy.json"
    policy_path = os.path.join(POLICY_DIR, f"{policy_name}.json")
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump({"url": request.url, "content": sanitize_content(content)}, f, ensure_ascii=False, indent=4)

    return {"message": f"Policy '{request.policy_name}' saved successfully."}

@app.post("/test")
def test_compliance(request: ComplianceRequest, authorization: str = Header(...)):
    if authorization != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")

    # Create results directory if not exists
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Extract website name from URL
    parsed_url = urlparse(request.url)
    website_name = parsed_url.netloc.replace('www.', '').split('.')[0]
    result_filename = os.path.join(RESULTS_DIR, f"result_{website_name}.json")

    # Check if result already exists
    if os.path.exists(result_filename):
        with open(result_filename, "r", encoding="utf-8") as f:
            return json.load(f)

    # Existing policy and content fetching logic
    policy_files = os.listdir(POLICY_DIR)
    if not policy_files:
        raise HTTPException(status_code=404, detail="No policies found. Train a policy first.")
    
    policy_path = os.path.join(POLICY_DIR, policy_files[0])
    
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)

    content = fetch_and_process_content(request.url) 
    structured_content = llm_parsing(mode="testing", raw_content=content, policy=policy)
    
    findings = structured_content["content"]  

    result = {
        "url": request.url,
        "llm_findings": findings,
        "violation_count": len(findings.split("\n")) 
    }

    # Save result to file
    with open(result_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return result