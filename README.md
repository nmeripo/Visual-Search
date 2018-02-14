### Requirement:
* Python 3.5

### Python Dependancies:
* scikit-learn 0.19.0
* numpy 1.22.1

* build_models.py ==> builds pretrained models, data dictionary and saves them in pickle format
* utils.py ==> has timeit method for function call time, elbow method for k-means clustering
* server.py ==> Flask API server serving HTTP POST requests

### POST REQUEST
* URL: http://127.0.0.1:5000/visual_search

* Example query: 
  - {"item_id": "131971", "query_type": "diverse"} in json format
  - query_type == "similar" for similar results
  - query_type == "diverse" for similar but diverse results

### Using python requests
```python
  r = requests.post("http://127.0.0.1:5000/predict", headers={'Content-Type': 'application/json'}, data=json.dumps({"item_id": "131971", "query_type": "diverse"}))
  print(r.content, r.status_code, r.reason)
  b'"{\\"diverse\\": \\"[460802, 30901, 176849, 157305, 452034, 244281, 269178, 207396, 96166, 243109]\\", \\"item_id\\": \\"131971\\"}"\n' 200 OK
```

### Using Postman Chrome extension
1. Enter POST request URL 
2. Choose Body -->  application/json --> raw -->  Enter example query --> Send
