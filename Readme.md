# Text Similarity from custom dataset
Getting textual similarities between invoice text files from a Doc2Vec model trained on the invoice dataset.

- **Input**:  invoice files in txt format.
- **Output**: SImilarities between the documents.


## Project Directory
```bash
├── app.py
├── data
│   ├── 20news-bydate-test
│   ├── 20news-bydate-test2
│   └── 20news-bydate-train
├── Docsim.py
├── document_similarity_finder.py
├── evaluate.py
├── init.py
├── models/
├── Readme.md
├── requirements.txt
├── static
│   └── ind.css
├── templates
│   └── index.html
├── train.py
└── utils.py
```




## Dependencies
Install the dependencies from the requirements.txt file
```console
pip install -r ./requirements.txt

```

## Train the model
- in train.py : 
At first, give the location of the dataset

```py
TEXT_DATA_DIR = "data/20news-bydate-train"

```

Then pass the doc2vec model with appropriate parameters, with the data path set before.
```py
d2v_model = trainDoc2Vec(TEXT_DATA_DIR, vector_size = 100, alpha = 0.025, min_count = 1, epochs=100,  save = True)

```
Finally, run the train.py file for training.
```console
python  train.py

```

## Inference
- in document_similarity_finder.py:
Give path of the doc2vec model and enter the two text.
```py
model_path_d2v = "models/d2v_Jun-15-2022"

```
The contents of the files will be passed to the models.

Run the document_similarity_finder.py file for finding the text similarity between the two given documents.

```console
python  document_similarity_finder.py

```

## Evaluate
- Generate csv for evaluation by running evaluate.py

## Start Flask server
- Run app.py 
