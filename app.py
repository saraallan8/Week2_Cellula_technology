import streamlit as st 
import torch 
import torch.nn as nn
from imagecaption import generate_caption
from PIL import Image
import pandas as pd
import os

def save_result(result_type, input_text, prediction, probability):
    result = {
        "result_type": result_type,
        "input_text": input_text,
        "prediction": prediction,
        "probability": probability
    }

    df = pd.DataFrame([result])

    file_path = os.path.join(os.path.dirname(__file__), "results.csv")

    df.to_csv(
        file_path,
        mode="a",
        header=not pd.io.common.file_exists(file_path),
        index = False
    )

def tokenize(text):
    return text.lower().split()

#Load the saved LSTM model
checkpoint = torch.load(
    "lstm_model.pt",
    map_location="cpu"
)

vocab = checkpoint["vocab"]
embedding_dim = checkpoint["embedding_dim"]
hidden_size = checkpoint["hidden_size"]
max_length = checkpoint["max_length"]

# Create the LSTM model

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

        self.fc = nn.Linear(
            hidden_size * 2,
            1
        )

    def forward(self, x):

        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        x = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1
        )

        output = self.fc(x)

        return output

model = LSTMClassifier(
    len(vocab),
    embedding_dim,
    hidden_size
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()

#Encode the text that user will enter it 

def encode(text):
    words = tokenize(text)

    numbers=[]

    for word in words:
        if word in vocab:
            numbers.append(vocab[word])
        else:
            numbers.append(vocab["<UNK>"])

    return numbers

max_length = 200

def pad_sequence(numbers):
    if len(numbers) < max_length:
        numbers = numbers + [vocab["<PAD>"]] * (max_length - len(numbers))
    else:
        numbers = numbers[:max_length]

    return numbers

def predict_text(text):

    numbers = encode(text)
    numbers = pad_sequence(numbers)
    input_tensor = torch.tensor(
        [numbers],
        dtype = torch.long
    )

    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

    if probability >= 0.9:
        prediction = "Toxic"
    else:
        prediction = "Non-toxic"     

    return prediction, probability
       
#streamlit Interface
st.title("🎭Toxic Comment Detector")

st.write("Type a comment to check if it's toxic or not")

#inputs and classification 

with st.form("classification_form", clear_on_submit=True):

    text = st.text_area("📝Enter your comment:")

    image = st.file_uploader(
        "🖼️ Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    classify_button = st.form_submit_button("Classify")

    if classify_button:

        if text.strip() != "":
            prediction, probability = predict_text(text)

            st.write("Prediction:", prediction)
            st.write("Toxic probability:", f"{probability:.2f}")

            save_result("Text", text, prediction, probability)

        if image is not None:
            image = Image.open(image)

            caption = generate_caption(image)

            st.write("Image caption:", caption)

            prediction, probability = predict_text(caption)

            st.write("Prediction:", prediction)
            st.write("Toxic probability:", f"{probability:.2f}")

            save_result("Image", caption, prediction, probability)

        if text.strip() == "" and image is None:
            st.warning("Please enter a comment or upload an image.")

st.subheader("📊 Classification Results")

if "show_results" not in st.session_state:
    st.session_state.show_results = False

if st.session_state.show_results:

    if st.button("Hide Results"):
        st.session_state.show_results = False
        st.rerun()

    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results.csv"
    )

    if os.path.exists(file_path):
        results = pd.read_csv(file_path)
        st.dataframe(results, use_container_width=True)
    else:
        st.write("No results stored yet.")

else:

    if st.button("Show Results"):
        st.session_state.show_results = True
        st.rerun()