from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

llm=ChatGroq(model="llama-3.1-8b-instant")
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser

class FieldsExtraction(BaseModel):
    findings:str=Field(description="Radiologist's technical observations")
    clinicaldata:str=Field(description="Reason for examination (e.g., symptoms like chest pain, shortness of breath)")
    ExamName:str=Field(description="Exam type and date")
    impression:str=Field(description="Final diagnosis or summary")

def llm_prediction(sentence):

    output_parser=PydanticOutputParser(pydantic_object=FieldsExtraction)
    format_instructions = output_parser.get_format_instructions()
    output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful medical data extraction assistant. 

                From the given "Report Text", extract the following fields and return ONLY a JSON object, and nothing else. 
                Use the format described in {format_instructions}.
                Return exactly this structure:
            
                Only return a JSON object with the following fields:

                ### Fields to be extracted:
                - findings
                - clinicaldata
                - ExamName
                - impression

                ### Example:

                Input:
                EXAM: CHEST RADIOGRAPHY EXAM DATE: 06/01/2019 08:30 PM. CLINICAL HISTORY: Cough. COMPARISON: None. TECHNIQUE: 2 views. FINDINGS: Lungs/Pleura: No focal opacities evident. No pleural effusion. No pneumothorax. Normal volumes. Mediastinum: Heart and mediastinal contours are unremarkable. Other: None. IMPRESSION: Normal 2-view chest radiography Dictated by: [[PERSONALNAME]] on 06/01/2019 08:42 PM. Electronically signed by: [[PERSONALNAME]] on 06/01/2019 08:43 PM.

                Extracted:
                findings = FINDINGS: Lungs/Pleura: No focal opacities evident. No pleural effusion. No pneumothorax. Normal volumes. Mediastinum: Heart and mediastinal contours are unremarkable. Other: None.  
                clinicaldata = CLINICAL HISTORY: Cough.  
                ExamName = EXAM: CHEST RADIOGRAPHY EXAM DATE: 06/01/2019 08:30 PM. TECHNIQUE: 2 views. COMPARISON: None.  
                impression = IMPRESSION: Normal 2-view chest radiography    Dictated by: [[PERSONALNAME]] on 06/01/2019 08:42 PM. Electronically signed by: [[PERSONALNAME]] on 06/01/2019 08:43 PM.
            
                ### NOTE: Return all fields as flat strings - do not nest them or break into subfields like 'examdate' or 'examname'. Return 'ExamName' as a single string, just as found in the report. """),
                    
                    
                ("user", "{input}") ])

    chain = prompt | llm | output_parser

    result=chain.invoke({"input":sentence, "format_instructions": output_parser.get_format_instructions()})

    return result
