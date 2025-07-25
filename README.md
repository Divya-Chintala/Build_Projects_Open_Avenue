# Build_Projects_Open_Avenue

## Fields Extraction from Radiology Reports

Medical NLP application designed to extract structured information from unstructured radiology report text using two approaches:
- Prompt-based extraction using Large Language Model -> Llama-3.1-8B-Instant
- Fine-tuned transformer model -> Bio_ClinicalBERT

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
