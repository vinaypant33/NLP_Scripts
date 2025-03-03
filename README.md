Named Entity Recognition (NER) - Supervised Learning Approach

  

This project is part of an ongoing research initiative at Dublin Business School (DBS), focusing on Named Entity Recognition (NER) using supervised learning with spaCy.

📌 Project Overview

The project implements a custom NER model trained using a supervised learning approach, requiring labeled data for identifying entities such as diseases and chemicals in medical texts. The model is trained and evaluated using spaCy's machine learning pipeline.

🚀 Features

✅ Supervised Learning-Based NER Model

Uses manually labeled training data for high accuracy.

Fine-tunes spaCy's transformer-based model.

✅ Custom NER Training Pipeline

Extracts entities from structured text.

Trains the model using spaCy's training loop.

✅ Model Evaluation with Confusion Matrix

Visualizes classification performance.

✅ NER Output with Colored Entity Highlighting

Displays detected entities in formatted output.

⚙️ Installation

1. Clone the Repository

   git clone https://github.com/your-username/ner-supervised-learning.git  
   cd ner-supervised-learning  

2. Install Dependencies

   pip install -r requirements.txt  

3. Download spaCy’s English Model

   python -m spacy download en_core_web_sm  

🛠️ Usage

1. Running the Model in Jupyter Notebook

Open and execute ner_project.ipynb to:✅ Load and preprocess data (from training.txt and test.txt).✅ Train the model on labeled entities.✅ Evaluate the model using a confusion matrix.✅ Visualize Named Entity Recognition (NER) results on sample text.

2. Testing the Model on Sample Text

The trained model can be tested on custom text input:

Input: "Aspirin is often used to treat headaches and fever."
Output: "Aspirin" → [CHEMICAL]
         "fever" → [DISEASE]

📊 Model Evaluation

The confusion matrix below summarizes the model’s classification performance:

      Pred: Disease   Pred: Chemical  
True: Disease    XX             XX  
True: Chemical   XX             XX  

🤝 Contributing

This project is part of a group research initiative at DBS. Contributions include:

Data annotation & preprocessing

Model training & evaluation

Performance optimization

If you'd like to contribute, feel free to submit a pull request! 🎯

📜 License

This project is for academic research purposes at Dublin Business School (DBS).


