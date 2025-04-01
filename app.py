import os,re
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException  
from pydantic import BaseModel
from Constants import FUNCTION_MAPPINGS
app = FastAPI()
from dotenv import load_dotenv
import logging
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


class QuestionRequest(BaseModel):
    question: str

@app.post("/api/")
async def get_answer(question: str = Form(...), file: UploadFile = File(None)):
    try:
        # Log the incoming request
        logger.info("Received question: %s", question)
        if file:
            logger.info("Received file: %s", file.filename)
        
        # Placeholder logic (to be replaced with actual LLM-based answer generation)
        function_to_run = find_best_matching_function(question, FUNCTION_MAPPINGS)
        answer = function_to_run  # Replace with actual logic
        
        # Log the generated answer
        logger.info("Generated answer: %s", answer)
        
        return {"answer": answer}
    
    except KeyError as e:
        logger.error("KeyError: %s", str(e))
        # raise HTTPException(status_code=400, detail=f"Invalid question: {str(e)}")
    
    except Exception as e:
        logger.error("An unexpected error occurred: %s", str(e))
        # raise HTTPException(status_code=500, detail="An unexpected error occurred.")

def extract_keywords(question):
    """Extract keywords from a question by cleaning and tokenizing."""
    cleaned = re.sub(r'[^\w\s-]', ' ', question.lower())
    keywords = re.findall(r'\b[\w/-]+\b', cleaned)
    return set(keywords)

def match_question_to_function(question, mappings):
    """Match a question to the most appropriate function based on keyword presence."""
    question_text = question.lower()
    max_matches = 0
    best_function = 'unknown_function'

    for mapping in mappings:
        matches = 0
        for keyword in mapping['keywords']:
            # Check if the keyword (as a whole phrase) exists in the question
            if keyword.lower() in question_text:
                matches += 1
        if matches > max_matches:
            max_matches = matches
            best_function = mapping['function']
    return best_function


def find_best_matching_function(question, mappings):
    # Flatten the keyword lists into a list of strings
    keyword_texts = [" ".join(mapping['keywords']) for mapping in mappings]
    
    # Add the user question to the corpus
    corpus = keyword_texts + [question]
    
    # Convert text into TF-IDF feature vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Compute cosine similarity between the question and each set of keywords
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Find the index of the highest similarity score
    best_match_index = similarity_scores.argmax()
    
    return mappings[best_match_index]['function']


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


