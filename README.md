# Build_Projects_Open_Avenue

## Fields Extraction from Radiology Reports

Medical NLP application designed to extract structured information from unstructured radiology report text using two approaches:
- Prompt-based extraction using Large Language Model -> Llama-3.1-8B-Instant
- Fine-tuned transformer model -> Bio_ClinicalBERT

#### Demo - [Extract fields from report text](https://buildprojectsopenavenue-wkdcukdq7bsuxrrffjxtex.streamlit.app/)

##### Sample input:

Exam: XR CHEST AP OR PA ONLY INDICATION: Dyspnea TECHNIQUE: Single view chest radiograph submitted for interpretation. COMPARISON: Yesterday FINDINGS: Persistent mild pulmonary vascular congestion, bibasilar airspace disease, and a trace left effusion. Interval removal of the endotracheal tube and the enteric tube. Otherwise, given differences in technique, the visualized portions of the remaining radiopaque support lines/tubes demonstrate no significant interval change in position since prior exam. No evidence of pneumothorax on this non-upright exam. IMPRESSION: 1. Persistent mild pulmonary vascular congestion, bibasilar airspace disease, and a trace left effusion. 2. Lines and tubes, as described above.

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
