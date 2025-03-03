**Steps Followed in the Code**

1. **Importing Required Libraries** - Load necessary Python libraries like `pandas`, `spacy`, and `re`.
2. **Loading Dataset** - Read `training.txt` and `test.txt` files.
3. **Splitting Data into Training and Testing Sets** - Process raw text into structured lists.
4. **Extracting Titles and Articles** - Use regex to extract relevant text.
5. **Processing Training and Testing Data** - Prepare structured data for training.
6. **Converting Data to Pandas DataFrame** - Store processed data in a tabular format.
7. **Extracting Labels for Named Entity Recognition** - Identify entity positions and labels.
8. **Preparing Data for Training** - Ensure dataset consistency and length matching.
9. **Converting Data to `spacy` Binary Format** - Transform data into `spaCy` format for training.
10. **Training the NER Model** - Train a Named Entity Recognition model using `spaCy`.
11. **Evaluating the Model** - Load and assess the trained model.
12. **Computing Evaluation Metrics** - Use confusion matrix and classification report for performance assessment.

### Importing Required Libraries
```python
import os
import pandas as pd
import re
import spacy
from spacy import logger
from spacy import displacy
from time import sleep
import warnings
warnings.filterwarnings("ignore")
```
- `os`: Provides functionality to interact with the operating system.
- `pandas as pd`: Used for data manipulation and analysis.
- `re`: Regular expressions for pattern matching.
- `spacy`: NLP library for Named Entity Recognition (NER).
- `logger`: Logging module from `spacy` for debugging.
- `displacy`: Visualization tool from `spacy`.
- `sleep`: Introduces delay (though not used in the code).
- `warnings.filterwarnings("ignore")`: Suppresses warnings.

### Loading Dataset
```python
with open("training.txt", "r") as train_file:
    training_data = train_file.read()

with open("test.txt", "r") as test_file:
    testing_data = test_file.read()
```
- Opens and reads `training.txt` and `test.txt` into `training_data` and `testing_data`, respectively.

### Splitting Data into Training and Testing Sets
```python
ctr = 0
train = []
for line in training_data.split("\n\n"):
  train.append(line)

ctr = 0
test = []
for line in testing_data.split("\n\n"):
  test.append(line)
```
- Splits the dataset based on double newline (`\n\n`) into `train` and `test` lists.

### Extracting Titles and Articles
```python
try:
  def article_extractor(text):
    article = re.findall(r'a\|(.*)\n' , text)
    return article[0]

  def title_extractor(text):
    title = re.findall(r't\|(.*)\n' , text)
    return title[0]
except Exception as e:
  pass
```
- Uses regex to extract articles (`a|...`) and titles (`t|...`).

### Processing Training and Testing Data
```python
train_article  = []
try:
  for x in train:
    train_article.append(title_extractor(x)+' '+article_extractor(x))

  for x in test:
    test_article.append(title_extractor(x)+' '+article_extractor(x))
except Exception as e:
  pass
```
- Extracts and combines titles and articles for both training and test datasets.

```python
test_article  = []
try:
  for x in test:
    test_article.append(title_extractor(x)+' '+article_extractor(x))
except Exception as e:
  pass
```
- Ensures `test_article` is initialized before use.

### Converting Data to Pandas DataFrame
```python
train_df  = pd.DataFrame(train_article, columns=['article'])
test_df  = pd.DataFrame(test_article, columns=['article'])
```
- Converts processed training and testing articles into Pandas DataFrames.

### Extracting Labels for Named Entity Recognition
```python
try:
  def get_labels(text):
    l  = re.findall(r'\t(.*)' , text)
    l = [x.split('\t') for x in l]
    labels  = []
    for i in l:
      try:
        labels.append((int(i[0]) , int(i[1]) , i[3]))
      except Exception as e:
        pass
    return labels

  def get_labels_and_entity(text):
    l = re.findall(r'\t(.*)' , text)
    l = [x.split('\t') for x in l]
    labels  = []
    for i in l:
      try:
        labels.append((int(i[0]) , int(i[1]) , i[3], i[2]))
      except Exception as e:
        pass
    return labels
except Exception as e:
  pass
```
- Extracts named entity labels from the dataset.
- `get_labels`: Extracts label positions and names.
- `get_labels_and_entity`: Also includes entity type.

```python
train_labels  = [get_labels(x) for x in train]
test_labels = [get_labels(x) for x in test]
```
- Creates lists of entity labels for training and test data.

### Preparing Data for Training
```python
min_length = min(len(test_df), len(test_labels), len(train_df), len(train_labels))
test_df = test_df.iloc[:min_length].copy()
train_df = train_df.iloc[:min_length].copy()
test_df['labels'] = test_labels[:min_length]
train_df['labels'] = train_labels[:min_length]
```
- Ensures that training and testing datasets have the same length.

### Converting Data to `spacy` Binary Format
```python
import spacy
import spacy.training
from spacy.tokens import DocBin

nlp = spacy.load("en_core_web_sm")
db  = DocBin()
```
- Loads `spaCy` model and prepares `DocBin` for storing training data.

```python
for text, annotations in training_data:
  doc  = nlp(text)
  ents = []
  for start, end , label in annotations:
    span  = doc.char_span(start , end , label=label)
    if not span == None:
      ents.append(span)
  doc.ents  = ents
  db.add(doc)
```
- Converts annotations into `spaCy`'s format.
- Stores processed data in `DocBin`.

```python
db.to_disk('train.spacy')
```
- Saves training data in binary format.

### Training the NER Model
```python
!python -m spacy init config base_config.cfg --lang en --pipeline ner
!python -m spacy init fill-config 'base_config.cfg' 'config.cfg'
!python -m spacy train config.cfg --output 'trained_output' --paths.train '/content/train.spacy' --paths.dev '/content/dev.spacy'
```
- Initializes and trains the NER model.

### Evaluating the Model
```python
import spacy
from spacy import displacy
nlp = spacy.load('/content/trained_output/model-best')
doc = nlp(testing_data[0][0])
spacy.displacy.render(doc ,style  = "ent" , jupyter = True)
```
- Loads the trained model and visualizes results.

### Computing Evaluation Metrics
```python
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for NER Model")
plt.show()
```
