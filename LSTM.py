import pandas as pd 
import torch
import torch.nn as nn 
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
import copy 

#Load the datset 
df = pd.read_csv("train.csv")

print(df.head())
print(df.columns)
print(df.shape)

#To use all column for the predict we need new column from all the column called label 
label_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

df["label"] = df[label_columns].max(axis=1)
print(df[["comment_text","label"]].head(10))

#Separate X And Y

x = df["comment_text"]
y = df["label"] 

print(x.head())
print(y.head())

#split the data into Training and Testing 
# First: train and test

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

# Second: split part of training into validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.1,
    random_state=42
)

print("train_size:", len(x_train))
print("validation_size:", len(x_val))
print("test_size:", len(x_test))

#tokenize the sentences into words 

def tokenize(text):
    return text.lower().split()

#Vocab to put all words we tokenize in one vocab 

word_count = Counter()

for text in x_train:
    words = tokenize(text)
    word_count.update(words)

print("Number of different words:", len(word_count))    

vocab = {
    "<PAD>":0,
    "<UNK>":1
}

for word, count in word_count.most_common(50000):
    vocab[word] = len(vocab)

print("vocab size:",len(vocab))    


#Encode to give the words numbers so the model can read them

def encode(text):
    words = tokenize(text)

    numbers = []

    for word in words:
        if word in vocab:
            numbers.append(vocab[word])
        else:
            numbers.append(vocab["<UNK>"])

    return numbers

#padding to make all words in the same length 

max_length = 200

def pad_sequence(numbers):
    if len(numbers) < max_length:
        numbers = numbers + [vocab["<PAD>"]] * (max_length - len(numbers))
    else:
        numbers = numbers[:max_length] 

    return numbers       


test = encode("I hate this movie!")

#And know we have to try this on all comments 

x_train_encoded = [pad_sequence(encode(text)) for text in x_train]

x_val_encoded = [pad_sequence(encode(text)) for text in x_val]

x_test_encoded = [pad_sequence(encode(text)) for text in x_test]

#tesors for pytorch

x_train_tensor = torch.tensor(
    x_train_encoded,
    dtype=torch.long
)

x_val_tensor = torch.tensor(
    x_val_encoded,
    dtype=torch.long
)

x_test_tensor = torch.tensor(
    x_test_encoded,
    dtype=torch.long
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val.values,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
)

#Create the LSTM Model

class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            hidden_size * 2,
            1
        )

    def forward(self, x):

        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        x = torch.cat((hidden[-2], hidden[-1]), dim=1)

        x = self.dropout(x)

        output = self.fc(x)

        return output

#Create the model 
embedding_dim = 300
hidden_size = 128

model = LSTMClassifier(
    len(vocab),
    embedding_dim,
    hidden_size
)

#Training the model
#1-DataLoader: gives the model small batches

train_dataset = TensorDataset(
    x_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    x_val_tensor,
    y_val_tensor
)

test_dataset = TensorDataset(
    x_test_tensor,
    y_test_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

#2.Loss Function:tells us how wrong the model is

positive = y_train.sum()
negative = len(y_train) - positive

pos_weight = negative / positive

print("Positive:", positive)
print("Negative:", negative)
print("Pos weight:", pos_weight)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
)

#3.Optimizer:changes the model's weights

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

#4.Training loop — repeats everything

epochs = 7

best_val_loss = float("inf")
best_model = None

for epoch in range(epochs):

    model.train()

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs.squeeze(), y_batch)

        loss.backward()

        optimizer.step()

    # validation
    model.eval()

    val_loss = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            outputs = model(x_batch)

            loss = criterion(
                outputs.squeeze(),
                y_batch
            )

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    print(
        "Epoch:", epoch + 1,
        "Train Loss:", loss.item(),
        "Validation Loss:", val_loss
    )

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model.state_dict())



model.load_state_dict(best_model)

print("Best validation loss:", best_val_loss)

# F1 evaluation
model.eval()

all_probabilities = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:

        outputs = model(X_batch)

        probabilities = torch.sigmoid(outputs.squeeze())

        all_probabilities.extend(probabilities.tolist())
        all_labels.extend(y_batch.tolist())

for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:

    predictions = [
        1 if p >= threshold else 0
        for p in all_probabilities
    ]

    f1 = f1_score(all_labels, predictions)

    print("Threshold:", threshold, "F1:", f1)


torch.save({
    "model_state": model.state_dict(),
    "vocab": vocab,
    "embedding_dim": embedding_dim,
    "hidden_size": hidden_size,
    "max_length": max_length
}, "lstm_model.pt")

print("Model saved!")    