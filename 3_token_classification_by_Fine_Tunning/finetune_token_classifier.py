from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "emilyalsentzer/Bio_ClinicalBERT"

label_list = ['O', 'B-ExamName', 'I-ExamName', 'E-ExamName', 'S-ExamName',
              'B-clinicaldata', 'I-clinicaldata', 'E-clinicaldata', 'S-clinicaldata',
              'B-findings', 'I-findings', 'E-findings', 'S-findings',
              'B-impression', 'I-impression', 'E-impression', 'S-impression']

label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

def tokenize_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    labels = []
    word_ids = tokenized_inputs.word_ids()  # Match subwords to word-level labels
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)  # Mask out
        elif word_idx != previous_word_idx:
            labels.append(label2id[example["labels"][word_idx]])
        else:
            # Same word → continuation subtoken
            labels.append(label2id[example["labels"][word_idx]])
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=17, id2label=id2label, label2id=label2id)

# Load preprocessed data
train_data = Dataset.from_json("train_merged.json")
test_data = Dataset.from_json("test_merged.json")
train_data = train_data.map(tokenize_labels)
test_data = test_data.map(tokenize_labels)

args = TrainingArguments(
    output_dir="./results/new",
    eval_strategy="epoch",           
    save_strategy="epoch",           
    load_best_model_at_end=True,     
    metric_for_best_model="f1",      
    greater_is_better=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs/new',
    logging_steps=10,
    seed=42
)


def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [[id2label[p] for (p, l) in zip(pred_row, label) if l != -100] for pred_row, label in zip(preds, labels)]
    
    # Flatten for F1
    y_true = [item for sublist in true_labels for item in sublist]
    y_pred = [item for sublist in true_preds for item in sublist]

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    #precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./results")
tokenizer.save_pretrained("./results")
print("Model saved to ./results")
