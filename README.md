# Named Entity Recognition (NER) - Supervised & Active Learning Approaches

This project is part of an ongoing research initiative at Dublin Business School (DBS), focusing on Named Entity Recognition (NER) using both **supervised learning** and **active learning** with spaCy.

## 📌 Project Overview

The project implements custom NER models trained using two different approaches:

- **Supervised Learning**: Requires manually labeled data for training a high-accuracy model.
- **Active Learning**: Reduces manual annotation effort by selecting uncertain samples for human labeling, improving efficiency.

Both models aim to identify entities such as diseases and chemicals in medical texts and are trained using spaCy's machine learning pipeline.

## 🚀 Features  

### ✅ **Supervised Learning-Based NER Model**  
- Uses fully labeled training data.
- Fine-tunes spaCy's transformer-based model.

### ✅ **Active Learning-Based NER Model**  
- Uses uncertainty sampling to suggest samples for human annotation.
- Reduces the need for complete manual labeling.
- Iteratively improves model accuracy with minimal human effort.

### ✅ **Custom NER Training Pipelines**  
- Extracts entities from structured text.
- Trains the model using spaCy's training loop.

### ✅ **Model Evaluation with Confusion Matrix**  
- Visualizes classification performance.

### ✅ **NER Output with Colored Entity Highlighting**  
- Displays detected entities in formatted output.

## ⚙️ Installation

### 1. Clone the Repository
```sh
git clone https://github.com/your-username/ner-supervised-learning.git  
cd ner-supervised-learning  
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt  
```

### 3. Download spaCy’s English Model
```sh
python -m spacy download en_core_web_sm  
```

## 🛠️ Usage

### 1. Running the Supervised Learning Model in Jupyter Notebook
Open and execute `ner_supervised.ipynb` to:  
- ✅ Load and preprocess data (from `training.txt` and `test.txt`).
- ✅ Train the model on labeled entities.
- ✅ Evaluate the model using a confusion matrix.
- ✅ Visualize Named Entity Recognition (NER) results on sample text.

### 2. Running the Active Learning Model in Jupyter Notebook
Open and execute `ner_active_learning.ipynb` to:  
- ✅ Load initial training data.
- ✅ Train the base model.
- ✅ Identify uncertain samples for annotation.
- ✅ Iteratively refine the model with human feedback.
- ✅ Evaluate performance improvements.

### 3. Testing the Models on Sample Text
The trained models can be tested on custom text input:

**Input:**  
```plaintext
"Aspirin is often used to treat headaches and fever."
```

**Output (NER Prediction):**  
```plaintext
"Aspirin" → [CHEMICAL]  
"fever" → [DISEASE]
```

## 📊 Model Evaluation & Comparison

Both models are evaluated using a confusion matrix summarizing classification performance:

|               | Pred: Disease | Pred: Chemical |
|--------------|--------------|--------------|
| **True: Disease**  | XX           | XX           |
| **True: Chemical** | XX           | XX           |

### 🔍 Key Differences

| Approach            | Pros                                           | Cons                                           |
|---------------------|---------------------------------|---------------------------------|
| **Supervised Learning** | High initial accuracy | Requires full dataset labeling |
| **Active Learning**   | Reduces manual labeling | Needs iterative human feedback |

## 🤝 Contributing

This project is part of a group research initiative at DBS. Contributions include:

- Data annotation & preprocessing
- Model training & evaluation
- Performance optimization

If you'd like to contribute, feel free to submit a pull request! 🎯

## 📜 License

This project is for academic research purposes at Dublin Business School (DBS).
