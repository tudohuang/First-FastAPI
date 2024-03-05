# First-FastAPI
My first FastAPI practice!

```bash
pip install torch tensorflow fastapi uvicorn 
```

## how to use?

method1:
```bash
python main.py
```
method2:
```bash
uvicorn main:app --reload
```

## Interaction:

run this when the server was ON!
```python
import requests

url = 'http://127.0.0.1:8000/predict/'
data = {"text": "your text here"}
response = requests.post(url, json=data)

print(response.json())
```


