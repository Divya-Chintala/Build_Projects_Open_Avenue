import streamlit as st
import pandas as pd
import numpy as np
from finetunnedBert.evaluateBert import bert_prediction
from prompting.evaluateLLm import llm_prediction



st.title(" Field Extraction from Radiology Reports ")

sentence = st.text_area("\n\n Enter Report Text to be classified :  \n\n")

model=st.selectbox("\n Chose model through which you wanted to classify the ReportText\n", ["Prompting with LLm", "Fine-Tunned Transformed"],index=None,placeholder="Select a model")

predict_key=st.button("Predict")

if predict_key:
    if model is not None and len(sentence)!=0:


        if "prompt" in model.lower() :
            st.markdown("LLM Prediction in progess....")
            

            merged=llm_prediction(sentence=sentence.lower())
            st.markdown("LLM via prompting has extracted below fields from give Report Text")
            merged = merged.dict()

            
        else:
            
            st.markdown("Bert Prediction in progess....")
            merged=bert_prediction(sentence=sentence)

            st.markdown("Finetunned Bio_ClinicalBert has extracted below fields from give Report Text")
        st.text_area("Report Text", value=sentence.lower(), height=200, disabled=True)
        result=pd.DataFrame(merged.items(),columns=["Fields Extracted", "Text"])
        result = result.sort_values(by="Fields Extracted")
        st.dataframe(result, use_container_width=True,row_height =90,width=300)



    elif len(sentence)==0:
        st.markdown("Please enter Report Text and then proceed to predict")
    else:
        st.markdown(" Please select the model and then proceed to predict")




       