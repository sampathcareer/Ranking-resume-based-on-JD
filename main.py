
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import shutil
from pydantic import BaseModel
from openai import OpenAI
from PyPDF2 import PdfReader
import os
import io
import re
from typing import List, Dict, Any
from pathlib import Path 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import pandas as pd
from fastapi import Form
import logging
from fastapi.responses import FileResponse

app = FastAPI()

GROQ_API_KEY = ""
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


# --- Constants ---
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class RankingCriterion(BaseModel):
    id: int
    criterion: str

class TransformedRankingCriteriaResponse(BaseModel):
    filename: str
    ranking_criteria: List[RankingCriterion]


def transform_ranking_criteria(criteria_string: str) -> List[RankingCriterion]:
    
    criteria_list = re.split(r'\n*\d+\.\s*', criteria_string.strip())
    criteria_list = criteria_list[1:]  
    ranking_criteria = []
    for i, criterion in enumerate(criteria_list):
        cleaned_criterion = criterion.strip()
        if cleaned_criterion: 
           ranking_criteria.append(RankingCriterion(id=i + 1, criterion=cleaned_criterion))
    return ranking_criteria

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def get_ranking_criteria(job_text):
    prompt = f"Extract the key ranking criteria from the following job description:\n\n{job_text}"
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "system", "content": "You are an AI that extracts job ranking criteria."},
                  {"role": "user", "content": prompt}],
        temperature=0.7, 
        max_tokens=1000 
    )
    return response.choices[0].message


@app.post("/upload-job-description/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith(".pdf"):
        job_text = extract_text_from_pdf(file_path)
    else:
        raise HTTPException(status_code=400, detail="only PDF files are allowed!")

    try:
        ranking_criteria = get_ranking_criteria(job_text)
        ranking_criteria_list = transform_ranking_criteria(ranking_criteria.content) # Parse the string into a list of objects


        return TransformedRankingCriteriaResponse(filename=file.filename, ranking_criteria=ranking_criteria_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract ranking criteria: {str(e)}")

    # return {"filename": file.filename, "ranking_criteria": ranking_criteria.content}


    # --- Helper Functions ---


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

def process_resume(file_path: str, criteria: List[str]) -> Dict[str, int]:
    
    file_extension = Path(file_path).suffix.lower()
    if file_extension == ".pdf":
        text = extract_text_from_pdf(file_path) 
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX files are supported.")

    text = clean_text(text)

    scores = {}
    for criterion in criteria:
        
        if criterion.lower() in text.lower():
            scores[criterion] = 5  
        else:
            scores[criterion] = 1 
    return scores


def extract_candidate_name(file_name: str) -> str:
     """Extracts a candidate name from the filename (basic implementation)."""
     name = Path(file_name).stem  # Remove extension
     name = name.replace("_", " ").replace("-", " ") # Replace underscores and hyphens
     return name 


# @app.post("/rank_resumes")
# async def rank_resumes(
#     criteria: List[str] = Form(...),  # Receive criteria as a list of strings from form data
#     files: List[UploadFile] = File(...)   # Receive files as a list of UploadFile objects
# ):

#     if not criteria:
#         raise HTTPException(status_code=400, detail="No ranking criteria provided.")

#     if not files:
#         raise HTTPException(status_code=400, detail="No resumes uploaded.")

#     candidate_data = []
#     # 1. Process each resume
#     for resume_file in files:
#         try:
#             file_path = os.path.join(UPLOAD_FOLDER, resume_file.filename)
#             with open(file_path, "wb") as f:
#                 shutil.copyfileobj(resume_file.file, f)

#             try:
#                 scores = process_resume(file_path, criteria)
#                 total_score = sum(scores.values())
#                 candidate_name = extract_candidate_name(resume_file.filename)  # Extract candidate name
#                 candidate_data.append({
#                     "Candidate Name": candidate_name,
#                     **scores,  # Spread the scores for each criterion
#                     "Total Score": total_score
#                 })


#             except ValueError as e:
#                 logging.error(f"Error processing resume {resume_file.filename}: {e}")
#                 raise HTTPException(status_code=400, detail=str(e))
#             except Exception as e:
#                 logging.error(f"Unexpected error processing resume {resume_file.filename}: {e}")
#                 raise HTTPException(status_code=500, detail=f"Error processing resume {resume_file.filename}: {e}")

#         except Exception as e:
#             logging.error(f"Error saving resume {resume_file.filename}: {e}")
#             raise HTTPException(status_code=500, detail=f"Error saving resume {resume_file.filename}: {e}")
#         finally:
#             await resume_file.close()

#     # 2. Create DataFrame from the collected data
#     df = pd.DataFrame(candidate_data)

#     # 3. Sort by Total Score
#     df = df.sort_values(by="Total Score", ascending=False)


#     # 4. Convert DataFrame to CSV
#     csv_file = io.StringIO()
#     df.to_csv(csv_file, index=False) # Exclude index column
#     csv_file.seek(0)  # Reset the buffer's position

#     # 5. Return CSV file as a StreamingResponse
#     return StreamingResponse(
#         iter([csv_file.getvalue()]),
#         media_type="text/csv",
#         headers={"Content-Disposition": "attachment;filename=resume_rankings.csv"}
#     )

def calculate_row_score(row: List[Any], criteria: List[str]) -> float:
    
    score = 0.0
    for criterion in criteria:
        try:
            
            criterion_index = row.index(criterion) 
            # Get the score from the row
            score_value = row[criterion_index+1] 
            
            if isinstance(score_value, (int, float)):
                score += float(score_value)
            elif isinstance(score_value, str):
                try:
                     score += float(score_value)
                except ValueError:
                    score += 0.0
            
        except (ValueError,IndexError):
            pass # ignore non-numeric values
    return score

def dataframe_to_rows(df: pd.DataFrame, criteria: List[str]):
    """
    Converts a Pandas DataFrame to a list of lists (rows), handling comma-separated
    values in cells by splitting them into separate rows and calculating scores.
    """
    rows = [df.columns.tolist()]  # Add header row
    for _, row in df.iterrows():
        row_data = row.tolist()
        new_rows = [[]]  

        for cell in row_data:
            if isinstance(cell, str) and "," in cell:
                
                values = cell.split(",")

                
                for i, value in enumerate(values):
                    if i >= len(new_rows):
                        new_rows.append([''] * len(row_data))  
                    new_rows[i].append(value.strip())  

                
                for i in range(len(row_data)):
                    if i < len(new_rows[0]):
                        pass  # Keep existing value
                    else:
                        new_rows[0].append(row_data[i])  


                rows.extend(new_rows)  
                new_rows = [[]]

            else:
                new_rows[0].append(cell)   

        
        current_row_data = [str(cell) for cell in new_rows[0]]
        score = calculate_row_score(current_row_data, criteria)
        new_rows[0].append(score) 

        rows.extend(new_rows)

    

    total_score = 0.0
    for i in range(1,len(rows)): #Skip header row
        try:
            score_index = len(rows[i]) - 1
            score = float(rows[i][score_index])
            total_score += score
        except (ValueError, IndexError):
            pass

    # rows.append(["Total Sum Score:",total_score])
    return rows

@app.post("/rank_resumes")
async def rank_resumes(
    criteria: List[str] = Form(...),  
    files: List[UploadFile] = File(...)  
):

    if not criteria:
        raise HTTPException(status_code=400, detail="No ranking criteria provided.")

    if not files:
        raise HTTPException(status_code=400, detail="No resumes uploaded.")

    candidate_data = []
   
    for resume_file in files:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, resume_file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(resume_file.file, f)

            try:
                scores = process_resume(file_path, criteria)
                total_score = sum(scores.values())
                candidate_name = extract_candidate_name(resume_file.filename)  
                candidate_data.append({
                    "Candidate Name": candidate_name,
                    **scores,  
                    "Total Score": total_score
                })


            except ValueError as e:
                logging.error(f"Error processing resume {resume_file.filename}: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logging.error(f"Unexpected error processing resume {resume_file.filename}: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing resume {resume_file.filename}: {e}")

        except Exception as e:
            logging.error(f"Error saving resume {resume_file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving resume {resume_file.filename}: {e}")
        finally:
            await resume_file.close()

   
    df = pd.DataFrame(candidate_data)

    
    df = df.sort_values(by="Total Score", ascending=False)

    
    rows = dataframe_to_rows(df, criteria) 

    
    csv_file = io.StringIO()

    
    for row in rows:
        csv_file.write(",".join(str(cell) for cell in row) + "\n")

    csv_file.seek(0)

    
    return StreamingResponse(
        iter([csv_file.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment;filename=resume_rankings.csv"}
    )