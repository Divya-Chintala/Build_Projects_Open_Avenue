

from metaflow import FlowSpec, step, Parameter, card
from IPython.display import display


import pandas as pd
import re
from dateutil import parser
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from matplotlib.figure import Figure

def fig_to_pil(fig: Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)





class MedicalNLPFlow(FlowSpec):

    open_data_path = Parameter("open_data_path", default="/content/drive/My Drive/Colab Notebooks/open_ave_data.csv")
    ground_truth_path = Parameter("ground_truth_path", default="/content/drive/My Drive/Colab Notebooks/ground_truth.csv")
    prompt_path = Parameter("prompt_path", default="/content/drive/My Drive/Colab Notebooks/prompt_output.csv")
    finetuned_path = Parameter("finetuned_path", default="/content/drive/My Drive/Colab Notebooks/finetuned_classified_output.csv")

    fields = ["findings", "clinicaldata", "ExamName", "impression"]


    # Load ground truth data and move to EDA.
    @step
    def start(self):
        print("Starting Medical NLP Flow")
        self.ground_truth_df = pd.read_csv(self.ground_truth_path)
        self.next(self.eda)

    @card
    @step
    def eda(self):
        print("EDA Step")

        # Load data
        data = pd.read_csv(self.open_data_path, header=0, index_col=0)

        print("Total no.of records in the dataset: ", data.shape[0])
        print("Total no.of columns in the dataset: ", data.shape[1])
        print("\nMissing values per column:\n", data.isnull().sum())

        null_clinicaldata = data[data['clinicaldata'].isnull()]
        print("\nRecords with missing clinicaldata:\n", null_clinicaldata)

        print("\nTotal duplicate rows in dataset:", data.duplicated().sum())
        print("Duplicate rows based on ReportText:", data['ReportText'].duplicated().sum())

        data = data.drop("ReportText", axis=1)

        # most frequent values
        dt = {}
        for col in data.columns:
            most_frequent = data[col].value_counts().index[0]
            freq = data[col].value_counts().iloc[0]
            dt[col] = {
                'Most Frequent Value': most_frequent,
                'Frequency': freq
            }
        df_freqs = pd.DataFrame(dt).T
        print("\nMost frequent values per column:\n", df_freqs)

        
        self.card = []

        # Bar chart of top 15 clinicaldata reasons
        data['clinical_reason'] = data['clinicaldata'].str.split(':').str[-1].str.lower().str.strip()
        data['clinical_reason'] = data['clinical_reason'].str.replace('.', '', regex=False)
        reason_counts = data['clinical_reason'].value_counts().head(15)
        data = data.drop("clinical_reason", axis=1)

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.barplot(y=reason_counts.index, x=reason_counts.values, ax=ax1)
        ax1.set_title("Most Frequent clinicaldata reasons")
        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("Clinical Reason")
        plt.tight_layout()
        self.card.append(fig_to_pil(fig1))
        plt.close(fig1)

        
        d = {}
        for col in data.columns:
            lengths = data[col].dropna().astype(str).map(len)
            d[col] = {
                "Min Length": lengths.min(),
                "Max Length": lengths.max(),
                "Avg Length": lengths.mean(),
                "Unique Entries": data[col].nunique(),
                "Duplicates count": data[col].duplicated().sum()
            }
        df_stats = pd.DataFrame(d).T
        print("\nText length and uniqueness statistics:\n", df_stats)

        # Sentence length histogram plots
        def plot_sentence_length_histogram(ax, text, title):
            text.dropna().astype(str).str.len().hist(ax=ax)
            ax.set_title(f"Sentence length of {title}")
            ax.grid(False)

        fig2, axes = plt.subplots(nrows=1, ncols=len(data.columns), figsize=(20, 4))
        for ax, col in zip(axes, data.columns):
            plot_sentence_length_histogram(ax, data[col], col)
        plt.tight_layout()
        self.card.append(fig_to_pil(fig2))
        plt.close(fig2)

        # WordCloud for findings
        stopwords = set(STOPWORDS)
        if 'findings' in data.columns:
            wordcloud = WordCloud(
                background_color='white',
                stopwords=stopwords,
                max_words=100,
                max_font_size=30,
                scale=3,
                random_state=1
            ).generate(str(data['findings']))

            fig3 = plt.figure(figsize=(12, 12))
            plt.axis('off')
            plt.imshow(wordcloud)
            plt.tight_layout()
            self.card.append(fig_to_pil(fig3))
            plt.close(fig3)

        # Save stats for later inspection
        df_freqs.to_csv("/content/drive/My Drive/Colab Notebooks/results/eda/eda_freq_summary.csv")
        df_stats.to_csv("/content/drive/My Drive/Colab Notebooks/results/eda/eda_text_stats.csv")
        self.df_freqs = df_freqs
        self.df_stats = df_stats



        self.next(self.prompting, self.finetuning)


    # model output generated via prompting
    @step
    def prompting(self):
        print(" Prompting Step")
        self.extracted_prompt = pd.read_csv(self.prompt_path)
        self.next(self.join)

    # model output from a fine-tuned model
    @step
    def finetuning(self):
        print("Fine-tuned Model Step")
        self.extracted_finetuned = pd.read_csv(self.finetuned_path)
        self.next(self.join)

    # Merge outputs from both prompting and finetuning steps 
    @step
    def join(self, inputs):
        print(" Joining Prompt and Finetune Results with Ground Truth")
        self.extracted_prompt = inputs.prompting.extracted_prompt
        self.extracted_finetuned = inputs.finetuning.extracted_finetuned
        self.ground_truth_df = inputs.prompting.ground_truth_df
        self.next(self.evaluate)

    # Compare each model's predictions with the ground truth and compute Exact Match Accuracy
    @step
    def evaluate(self):
        print(" Evaluating Exact Match Accuracy")

        # Cleaning and normalization functions
        # whitespaces, punctuation, lowercasing
        def clean_text(text):
            if not isinstance(text, str):
                return ""
            text = text.strip().replace('\xa0', ' ').replace('\u200b', '').lower()
            text = re.sub(r'\s*:\s*', ':', text)
            text = re.sub(r'\s*/\s*', '/', text)
            text = re.sub(r'\s*-\s*', '-', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s*\.\s*', '.', text)
            text = re.sub(r'\[\s+', '[', text)
            text = re.sub(r'\s+\]', ']', text)
            text = re.sub(r'\d{2}/\d{2}/\d{4} \d{1,2}:\d{2}(?: ?[ap]m| ?[AP]M)',
                          lambda m: parser.parse(m.group()).strftime("%m/%d/%Y %I:%M %p").lower(), text)
            return text

        def canonicalize_examname(text):
            if not isinstance(text, str): return ""
            parts = re.findall(r'(exam.*?)(comparison:.*?\.)?(technique:.*?\.)?', text)
            if parts:
                exam, comparison, technique = parts[0]
                return f"{exam}{technique or ''}{comparison or ''}".strip()
            return text

        def canonicalize_sentences(text):
            if not isinstance(text, str): return ""
            parts = [p.strip() for p in text.split('.') if p.strip()]
            return '. '.join(sorted(parts)) + '.'

        results = []

        for mode_name, df in [("Prompt", self.extracted_prompt.copy()), ("Finetuned", self.extracted_finetuned.copy())]:
            gt = self.ground_truth_df.copy()
            for field in self.fields:
                df[field] = df[field].apply(clean_text)
                gt[field] = gt[field].apply(clean_text)

                if field.lower() == "examname":
                    df[field] = df[field].apply(canonicalize_examname)
                    gt[field] = gt[field].apply(canonicalize_examname)

                if field.lower() in ["impression", "findings"]:
                    df[field] = df[field].apply(canonicalize_sentences)
                    gt[field] = gt[field].apply(canonicalize_sentences)

            for field in self.fields:
                exact_matches = (df[field] == gt[field]).sum()
                total = len(df)
                exact_accuracy = exact_matches / total * 100
                results.append({
                    "Model": mode_name,
                    "Field": field,
                    "Exact Match Accuracy (%)": round(exact_accuracy, 2)
                })

        self.evaluation_df = pd.DataFrame(results)
        self.next(self.end)

    # print and save the evaluation results
    @step
    def end(self):
        print("Flow Completed. Final Evaluation:\n")
        print(self.evaluation_df)
        self.evaluation_df.to_csv("/content/drive/My Drive/Colab Notebooks/results/final_metaflow_eval.csv", index=False)


if __name__ == '__main__':
    MedicalNLPFlow()
