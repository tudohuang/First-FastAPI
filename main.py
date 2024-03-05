from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# 加載Tokenizer配置
with open(r"tokenizer_config.json") as f:
    tokenizer_data = f.read()
tokenizer = tokenizer_from_json(tokenizer_data)

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# 模型參數
hidden_dim = 128
num_layers = 2
vocab_size = 10000  # 根據您的 tokenizer 配置調整
embed_dim = 100
num_classes = 6  # 分類類別的數量
max_seq_length = 171  # 序列的最大長度

model = TextLSTM(vocab_size, embed_dim, hidden_dim, num_layers, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(r"lstm.pt", map_location=device))
model.to(device)
model.eval()

# 預處理函數
def prodata(sentence):
    test_tokenized = tokenizer.texts_to_sequences([sentence])
    test_padded = pad_sequences(test_tokenized, maxlen=max_seq_length)
    test_padded_tensor = torch.tensor(test_padded, dtype=torch.long).to(device)
    return test_padded_tensor

# 預測函數
def greet(text):
    ptext = prodata(text)
    with torch.no_grad():
        prediction = model(ptext)
        predicted_class = torch.argmax(prediction, dim=1)
        result = predicted_class.item()
        emotions = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
        return emotions[result]

# FastAPI 應用
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
async def predict(text_input: TextInput):
    result = greet(text_input.text)
    return {"emotion": result}

# 啟動服務器（如果您是直接運行這個文件）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
