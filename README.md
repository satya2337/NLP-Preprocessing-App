# 🧠 NLP Preprocessing App (Streamlit)

This is a **Streamlit-based NLP web application** that allows users to perform core **text preprocessing techniques** such as Tokenization, Text Cleaning, Stemming, Lemmatization, and Bag of Words on any input text.

The app is designed with a modern dark UI and interactive tables & charts for better understanding of NLP concepts.

---

## 🚀 Features
- Tokenization(Sentence Tokenization, Word Tokenization , Character Tokenization)  
- Text Cleaning (lowercasing, punctuation & number removal, stopword removal)  
- Stemming using (Porter Stemmer ,Lancaster Stemmer)  
- Lemmatization using spaCy  
- Part-of-Speech (POS) tagging  
- Bag of Words (BoW) frequency table  
- Pie chart visualization of top 10 frequent words  
- Interactive Streamlit UI with sidebar controls  

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- NLTK  
- spaCy  
- Scikit-learn  
- Pandas  
- Matplotlib  

---

## 📦 Installation
Make sure Python is installed on your system.Then 

Install all required libraries:

pip install streamlit nltk spacy scikit-learn pandas matplotlib  

Download NLP models:

python -m spacy download en_core_web_sm  
python -m nltk.downloader punkt stopwords  
---

## ▶ How to Run the App

Go to the project folder and run:

streamlit run Project.py  

The application will open automatically in your web browser.

---

## 🧪 What This App Does

The user enters any text and selects one of the NLP techniques from the sidebar:

### Tokenization
Splits text into:
- Sentences  
- Words  
- Characters  

### Text Cleaning
- Converts text to lowercase  
- Removes punctuation and numbers  
- Removes stopwords using spaCy  
- Outputs cleaned text  

### Stemming
- Uses:
  - Porter Stemmer  
  - Lancaster Stemmer  
- Displays results in a comparison table  

### Lemmatization
- Uses spaCy to generate:
  - Word  
  - POS (Part of Speech)  
  - Lemma  
- Shows results in a dataframe  

### Bag of Words
- Uses CountVectorizer  
- Displays word frequencies in a table  
- Shows a pie chart for top 10 words  

---

## 📁 Project Structure

NLP-Preprocessing-App  
│  
├── Project.py  
├── README.md  
└── requirements.txt  

---

## 🎯 Use Cases

- Learning Natural Language Processing  
- Understanding text preprocessing  
- Preparing text for Machine Learning models  
- Educational and academic projects  

---

## 👨‍💻 Author

**Satya Anand**  
GitHub: https://github.com/satya2337
