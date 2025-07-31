# Build_Projects_Open_Avenue

## Fields Extraction from Radiology Reports

Medical NLP application designed to extract structured information from unstructured radiology report text using two approaches:
- Prompt-based extraction using Large Language Model -> Llama-3.1-8B-Instant
- Fine-tuned transformer model -> Bio_ClinicalBERT

#### Demo - [Extract fields from report text](https://buildprojectsopenavenue-wkdcukdq7bsuxrrffjxtex.streamlit.app/)

##### Sample input:

EXAM DESCRIPTION: X-ray single view chest. CLINICAL HISTORY: 68 years Male, SOB COMPARISON: None. TECHNIQUE: Single portable x-ray view of the chest performed on 06/02/2020 at 11:34 PM FINDINGS: The lungs are well expanded and are clear. There is no evidence of a pneumothorax. The cardiac silhouette is normal in size and configuration. The mediastinal contours are normal. No acute osseous abnormality is identified. No focal soft tissue abnormalities are seen. Lines and tubes: None. IMPRESSION: No evidence of acute intrathoracic disease. Electronically signed by: [[PERSONALNAME]] 06/02/2020 12:02 AM CDT Workstation: [ID]

---

## Project Overview

Radiology reports are rich in clinical insight but typically unstructured. This project enables structured field extraction (like Findings, ExamName, Clinicaldata, Impression) via:

- Prompting LLM with few shots
- Fine-tuned ClinicalBert for token classification using labeled medical text

Inferencing is done through streamlit interface, which allows users to compare both models interactively.

---

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Llama-3.1-8B-Instant
- **Transformer Model**: Fine-tuned Bio_ClinicalBERT 
- **Optional Orchestration**: Metaflow for training and evaluation
- **Frameworks**: Hugging Face Transformers, PyTorch, Pandas, scikit-learn

---

## Results

| Field         | Prompt Accuracy (%) | Fine-Tuned Accuracy (%) |
|---------------|---------------------|--------------------------|
| ExamName      | 86.27               | 83.02                    |
| ClinicalData  | 84.17               | 96.54                    |
| Findings      | 76.83               | 95.39                    |
| Impression    | 70.75               | 94.97                    |

**Insight**: Fine-tuned Bio_ClinicalBERT significantly outperforms prompting for critical fields like *Findings* and *Impression*, showing 18–24% gains in exact match accuracy.


---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/Divya-Chintala/Build_Projects_Open_Avenue.git
cd Build_Projects_Open_Avenue

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # MAC
venv\Scripts\activate # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app/app.py
