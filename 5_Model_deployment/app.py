import streamlit as st
import pandas as pd
from finetunnedBert.evaluateBert import bert_prediction
from prompting.evaluateLLm import llm_prediction
from st_aggrid import AgGrid, GridOptionsBuilder

st.title("Field Extraction from Radiology Reports")

sentence = st.text_area("\n\n Enter Report Text to be classified :  \n\n")

model = st.selectbox(
    "\n Choose model through which you want to classify the Report Text\n", 
    ["Prompting with LLm", "Fine-Tunned Bi_ClinicalBert", "Compare results of both"],
    index=None,
    placeholder="Select a model"
)

predict_key = st.button("Predict")

if predict_key:
    if model is not None and len(sentence) != 0:

        st.text_area("Report Text", value=sentence.lower(), height=200, disabled=True)

        if "prompt" in model.lower():
            st.markdown("LLM Prediction in progress....")
            merged = llm_prediction(sentence=sentence.lower()).dict()
            st.markdown("LLM via prompting has extracted the following fields:")
            
            result = pd.DataFrame(merged.items(), columns=["Fields Extracted", "Text"])
            result = result.sort_values(by="Fields Extracted")

            gb = GridOptionsBuilder.from_dataframe(result)
            gb.configure_default_column(wrapText=True, autoHeight=True)
            grid_options = gb.build()
            AgGrid(result, gridOptions=grid_options, height=300, fit_columns_on_grid_load=True)

        elif "bert" in model.lower():
            st.markdown("BERT Prediction in progress....")
            merged = bert_prediction(sentence=sentence)
            st.markdown(" Fine-tuned Bio_ClinicalBERT has extracted the following fields:")

            result = pd.DataFrame(merged.items(), columns=["Fields Extracted", "Text"])
            result = result.sort_values(by="Fields Extracted")

            gb = GridOptionsBuilder.from_dataframe(result)
            gb.configure_default_column(wrapText=True, autoHeight=True)
            grid_options = gb.build()
            AgGrid(result, gridOptions=grid_options, height=300, fit_columns_on_grid_load=True)

        elif "compare" in model.lower():
            st.markdown(" Comparing predictions from both models...")

          
            llm_output = llm_prediction(sentence=sentence.lower()).dict()
            bert_output = bert_prediction(sentence=sentence)

           
            all_keys = sorted(set(llm_output.keys()).union(set(bert_output.keys())))
            data = []
            for key in all_keys:
                data.append({
                    "Field": key,
                    "LLM Extracted Text": llm_output.get(key, ""),
                    "BERT Extracted Text": bert_output.get(key, "")
                })
            result = pd.DataFrame(data)

            st.markdown(" Side-by-side comparison of LLM and BERT predictions:")

            gb = GridOptionsBuilder.from_dataframe(result)
            gb.configure_default_column(wrapText=True, autoHeight=True)
            grid_options = gb.build()
            AgGrid(result, gridOptions=grid_options, height=400, fit_columns_on_grid_load=True)

    elif len(sentence) == 0:
        st.markdown("Please enter Report Text before predicting.")
    else:
        st.markdown("Please select a model before predicting.")
