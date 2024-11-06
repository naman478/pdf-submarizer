# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import PyPDF2
# import re
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os

# app = Flask(__name__)
# CORS(app)  # Enable CORS to allow requests from React frontend

# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def extract_text_from_pdf(pdf_file):
#     """Extract text from PDF file."""
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# def preprocess_text(text):
#     """Clean and preprocess the text."""
#     # Remove special characters and extra whitespace
#     text = re.sub(r'[^\w\s.]', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# def split_into_sentences(text):
#     """Split text into sentences."""
#     return [s.strip() for s in text.split('.') if s.strip()]

# def get_summary(text, num_sentences=5):
#     """Generate summary using TF-IDF and cosine similarity."""
#     # Preprocess text
#     clean_text = preprocess_text(text)
#     sentences = split_into_sentences(clean_text)
    
#     if len(sentences) <= num_sentences:
#         return ' '.join(sentences)

#     # Create TF-IDF matrix
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(sentences)

#     # Calculate similarity matrix
#     similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

#     # Calculate sentence scores
#     scores = similarity_matrix.sum(axis=1)
    
#     # Get top sentences
#     ranked_sentences = [sentences[i] for i in scores.argsort()[-num_sentences:][::-1]]
    
#     # Sort sentences by their original order
#     original_order = sorted(ranked_sentences, key=lambda x: sentences.index(x))
    
#     return '. '.join(original_order) + '.'

# def summarize_pdf(pdf_path, num_sentences=5):
#     """Main function to summarize PDF."""
#     try:
#         text = extract_text_from_pdf(pdf_path)
#         summary = get_summary(text, num_sentences)
#         return {"success": True, "summary": summary}
#     except Exception as e:
#         return {"success": False, "error": str(e)}

# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     if 'file' not in request.files:
#         return jsonify({"success": False, "error": "No file part"})

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"success": False, "error": "No selected file"})

#     if file and file.filename.endswith('.pdf'):
#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(file_path)

#         # Summarize the PDF
#         result = summarize_pdf(file_path)
#         return jsonify(result)

#     return jsonify({"success": False, "error": "Unsupported file type"})

# if __name__ == '__main__':
#     app.run(debug=True)

import numpy as np
import networkx as nx
import PyPDF2
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    """Clean and preprocess the text."""
    # Remove special characters and extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.]', '', text)
    return text.strip()

def lemmatize_text(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])

def textrank_summarize(text, num_sentences=5):
    """Generate summary using an advanced TextRank algorithm."""
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return text
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    
    # Lemmatize and clean sentences
    clean_sentences = [lemmatize_text(preprocess_text(sent.lower())) for sent in sentences]
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(clean_sentences)
    
    # Create similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Create network graph
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Get top sentences
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    
    # Select top N sentences and sort them by their original position
    top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: sentences.index(x[1]))
    
    # Compress sentences
    compressed_sentences = [compress_sentence(sent) for _, sent in top_sentences]
    
    summary = ' '.join(compressed_sentences)
    return summary

def compress_sentence(sentence):
    """Compress a sentence by removing less important words."""
    words = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    
    # Keep nouns, verbs, adjectives, and adverbs
    important_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'RB']
    compressed_words = [word for word, tag in pos_tags if tag in important_tags]
    
    return ' '.join(compressed_words)

def lsa_summarize(text, num_sentences=5):
    """Generate summary using LSA (Latent Semantic Analysis)."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summarizer.stop_words = stopwords.words('english')
    
    summary = summarizer(parser.document, num_sentences)
    return ' '.join(str(sentence) for sentence in summary)

def hybrid_summarize(text, num_sentences=5):
    """Generate summary using a hybrid approach."""
    # Use TextRank for extractive summarization
    textrank_summary = textrank_summarize(text, num_sentences)
    
    # Use LSA for another perspective on summarization
    lsa_summary = lsa_summarize(text, num_sentences)
    
    # Combine both summaries
    combined_summary = textrank_summary + " " + lsa_summary
    
    # Ensure the final summary doesn't exceed the desired length
    final_summary = ' '.join(sent_tokenize(combined_summary)[:num_sentences])
    
    return final_summary

def summarize_pdf(pdf_path, num_sentences=5):
    """Main function to summarize PDF."""
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Clean text
        clean_text = preprocess_text(text)
        
        # Generate summary using hybrid approach
        summary = hybrid_summarize(clean_text, num_sentences)
        
        return {"success": True, "summary": summary}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})

    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Get number of sentences from request (default to 5)
        num_sentences = request.form.get('num_sentences', 5, type=int)

        # Summarize the PDF
        result = summarize_pdf(file_path, num_sentences)
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        return jsonify(result)

    return jsonify({"success": False, "error": "Unsupported file type"})

if __name__ == '__main__':
    app.run(debug=True)

# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import PyPDF2
# import re
# from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import networkx as nx


# # Initialize Flask app and CORS
# app = Flask(__name__)
# CORS(app)

# try:
#     # Your existing NLTK code
#     import nltk
#     nltk.download('punkt')
# except Exception as e:
#     print("NLTK error:", e)


# # Download required NLTK data

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# class PDFSummarizer:
#     def __init__(self):
#         self.stop_words = set(stopwords.words('english'))
#         self.lemmatizer = WordNetLemmatizer()
        
#     def extract_text_from_pdf(self, pdf_file):
#         """Extract text from PDF file with improved handling."""
#         try:
#             reader = PyPDF2.PdfReader(pdf_file)
#             text = ""
#             for page in reader.pages:
#                 text += page.extract_text()
#             return text
#         except Exception as e:
#             raise Exception(f"Error extracting text from PDF: {str(e)}")

#     def preprocess_text(self, text):
#         """Enhanced text preprocessing."""
#         # Convert to lowercase
#         text = text.lower()
        
#         # Remove special characters and numbers
#         text = re.sub(r'[^\w\s.]', ' ', text)
        
#         # Remove extra whitespace
#         text = re.sub(r'\s+', ' ', text)
        
#         # Split into sentences
#         sentences = sent_tokenize(text)
        
#         # Clean and preprocess each sentence
#         cleaned_sentences = []
#         for sentence in sentences:
#             if len(sentence.strip()) > 10:  # Filter out very short sentences
#                 cleaned_sentences.append(sentence.strip())
        
#         return cleaned_sentences

#     def create_sentence_vectors(self, sentences):
#         """Create sentence vectors using TF-IDF."""
#         tfidf = TfidfVectorizer(
#             stop_words='english',
#             max_features=200,
#             max_df=0.95,
#             min_df=2
#         )
#         tfidf_matrix = tfidf.fit_transform(sentences)
#         return tfidf_matrix

#     def create_similarity_matrix(self, tfidf_matrix):
#         """Create similarity matrix using cosine similarity."""
#         similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
#         # Normalize the matrix
#         norm = similarity_matrix.sum(axis=1)
#         similarity_matrix_normalized = similarity_matrix / norm[:, np.newaxis]
        
#         return similarity_matrix_normalized

#     def textrank_scores(self, similarity_matrix):
#         """Calculate TextRank scores for sentences."""
#         nx_graph = nx.from_numpy_array(similarity_matrix)
#         scores = nx.pagerank(nx_graph)
#         return scores

#     def get_summary(self, text, num_sentences=5, diversity_penalty=0.7):
#         """Generate improved summary using TextRank and diversity penalty."""
#         # Preprocess text
#         sentences = self.preprocess_text(text)
        
#         if len(sentences) <= num_sentences:
#             return ' '.join(sentences)

#         # Create sentence vectors
#         tfidf_matrix = self.create_sentence_vectors(sentences)
        
#         # Create similarity matrix
#         similarity_matrix = self.create_similarity_matrix(tfidf_matrix)
        
#         # Calculate TextRank scores
#         scores = self.textrank_scores(similarity_matrix)
        
#         # Convert scores dictionary to list of tuples (index, score)
#         ranked_sentences = [(idx, scores[idx]) for idx in range(len(sentences))]
        
#         # Sort sentences by score
#         ranked_sentences.sort(key=lambda x: x[1], reverse=True)
        
#         # Select top sentences with diversity penalty
#         selected_sentences = []
#         selected_indices = []
        
#         for idx, score in ranked_sentences:
#             if len(selected_sentences) >= num_sentences:
#                 break
                
#             # Apply diversity penalty
#             if not selected_indices:
#                 selected_sentences.append(sentences[idx])
#                 selected_indices.append(idx)
#             else:
#                 # Calculate similarity with already selected sentences
#                 similarities = [similarity_matrix[idx][prev_idx] for prev_idx in selected_indices]
#                 max_similarity = max(similarities)
                
#                 # Apply penalty and check if sentence should be included
#                 penalized_score = score * (1 - diversity_penalty * max_similarity)
                
#                 if penalized_score > 0.1:  # Threshold to ensure quality
#                     selected_sentences.append(sentences[idx])
#                     selected_indices.append(idx)
        
#         # Sort selected sentences by their original position
#         selected_sentences = [x for _, x in sorted(zip(selected_indices, selected_sentences))]
        
#         # Join sentences and post-process
#         summary = ' '.join(selected_sentences)
        
#         # Final cleanup
#         summary = re.sub(r'\s+', ' ', summary)
#         summary = summary.strip()
        
#         return summary

#     def summarize_pdf(self, pdf_path, num_sentences=5):
#         """Main function to summarize PDF with error handling."""
#         try:
#             # Extract text from PDF
#             text = self.extract_text_from_pdf(pdf_path)
            
#             # Check if text was successfully extracted
#             if not text.strip():
#                 return {
#                     "success": False,
#                     "error": "No text could be extracted from the PDF"
#                 }
            
#             # Generate summary
#             summary = self.get_summary(text, num_sentences)
            
#             # Validate summary
#             if not summary.strip():
#                 return {
#                     "success": False,
#                     "error": "Could not generate a meaningful summary"
#                 }
            
#             return {
#                 "success": True,
#                 "summary": summary,
#                 "original_length": len(text.split()),
#                 "summary_length": len(summary.split())
#             }
            
#         except Exception as e:
#             return {
#                 "success": False,
#                 "error": str(e)
#             }

# # Define a route for summarizing PDFs
# @app.route("/upload", methods=["POST"])
# def summarize_pdf_route():
#     file = request.files.get("pdf_file")
#     num_sentences = int(request.form.get("num_sentences", 5))

#     if not file:
#         return jsonify({"success": False, "error": "No file provided"}), 400

#     summarizer = PDFSummarizer()
#     result = summarizer.summarize_pdf(file, num_sentences)
#     return jsonify(result)

# # Run the Flask app if this script is executed directly
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

