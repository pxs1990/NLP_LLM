import pandas as pd
from nlpaug.augmenter.word import ContextualWordEmbsAug
import numpy as np
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW

# Load the dataset
print("This will load the dataset")
df = pd.read_csv('movie_review_data.csv')

# Initialize the augmenter
aug = ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")

# Get the minority class samples
minority_class_samples = df[df['label'] == df['label'].value_counts().idxmin()]

# Augment the minority class twice
augmented_texts = []
for _ in range(2):  # Augmenting twice
    for text in minority_class_samples['text']:
        augmented_texts.append(aug.augment(text))

# Create a new DataFrame with the augmented texts and labels
augmented_df = pd.DataFrame(
    {'text': augmented_texts, 'label': [df['label'].value_counts().idxmin()] * len(augmented_texts)})

# Append the augmented data to the original DataFrame
df = pd.concat([df, augmented_df])

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Split the data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Tokenize the texts, ensuring each item is treated as a string
train_encodings = tokenizer([str(text) for text in train_texts], padding=True, truncation=True, max_length=128, return_tensors='pt')
val_encodings = tokenizer([str(text) for text in val_texts], padding=True, truncation=True, max_length=128, return_tensors='pt')

# Convert labels to tensor
train_labels = torch.tensor(train_labels.tolist())
val_labels = torch.tensor(val_labels.tolist())


# Define a custom dataset class
class MovieReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# Create a custom datasets
train_dataset = MovieReviewDataset(train_encodings, train_labels)
val_dataset = MovieReviewDataset(val_encodings, val_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)

# Load pre-trained BERT model with a classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
# this will optimize the model performance