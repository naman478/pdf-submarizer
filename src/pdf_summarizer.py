import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import networkx as nx
import os

class PDFSummarizer:
    def __init__(self):
        self._initialize_nltk()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def _initialize_nltk(self):
        """Initialize NLTK resources with proper error handling."""
        nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
        os.makedirs(nltk_data_path, exist_ok=True)
        
        required_resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for resource in required_resources:
            try:
                nltk.data.find(f'{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    raise Exception(f"Failed to download NLTK resource '{resource}': {str(e)}")

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file with improved handling."""
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            
            if len(reader.pages) > 5:
                raise ValueError("PDF exceeds the maximum limit of 5 pages")
                
            text = ""
            for page in reader.pages:
                text += page.extract_text()
                
            if not text.strip():
                raise ValueError("No text content found in the PDF")
                
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def preprocess_text(self, text):
        """Enhanced text preprocessing with better sentence handling."""
        # Convert to lowercase and basic cleaning
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        
        # Split into sentences with error handling
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback to basic sentence splitting if NLTK fails
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Clean and validate sentences
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = re.sub(r'[^\w\s.]', ' ', sentence)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Filter out invalid or too short sentences
            if len(cleaned.split()) >= 5:
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences

    def create_sentence_vectors(self, sentences):
        """Create sentence vectors using TF-IDF with robust parameters."""
        if len(sentences) < 2:
            return None
            
        try:
            tfidf = TfidfVectorizer(
                stop_words='english',
                max_features=200,
                max_df=0.95,
                min_df=1,
                ngram_range=(1, 2)
            )
            return tfidf.fit_transform(sentences)
        except Exception:
            return None

    def create_similarity_matrix(self, tfidf_matrix):
        """Create similarity matrix with error handling."""
        try:
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Normalize the matrix
            norm = similarity_matrix.sum(axis=1)
            similarity_matrix_normalized = similarity_matrix / np.where(norm[:, np.newaxis] != 0, 
                                                                     norm[:, np.newaxis], 
                                                                     1)
            
            return similarity_matrix_normalized
            
        except Exception:
            return None

    def textrank_scores(self, similarity_matrix):
        """Calculate TextRank scores with fallback options."""
        try:
            nx_graph = nx.from_numpy_array(similarity_matrix)
            return nx.pagerank(nx_graph, max_iter=200)
        except Exception:
            # Fallback to basic scoring if PageRank fails
            return {i: sum(row) for i, row in enumerate(similarity_matrix)}

    def get_summary(self, text, num_sentences=5, diversity_penalty=0.7):
        """Generate summary with improved robustness."""
        # Preprocess text
        sentences = self.preprocess_text(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)

        # Create sentence vectors
        tfidf_matrix = self.create_sentence_vectors(sentences)
        if tfidf_matrix is None:
            return ' '.join(sentences[:num_sentences])
        
        # Create similarity matrix
        similarity_matrix = self.create_similarity_matrix(tfidf_matrix)
        if similarity_matrix is None:
            return ' '.join(sentences[:num_sentences])
        
        # Calculate TextRank scores
        scores = self.textrank_scores(similarity_matrix)
        
        # Select sentences
        ranked_sentences = [(idx, score) for idx, score in scores.items()]
        ranked_sentences.sort(key=lambda x: x[1], reverse=True)
        
        selected_sentences = []
        selected_indices = []
        
        for idx, score in ranked_sentences:
            if len(selected_sentences) >= num_sentences:
                break
            
            # Apply diversity penalty
            if not selected_indices:
                selected_sentences.append(sentences[idx])
                selected_indices.append(idx)
            else:
                similarities = [similarity_matrix[idx][prev_idx] for prev_idx in selected_indices]
                max_similarity = max(similarities)
                penalized_score = score * (1 - diversity_penalty * max_similarity)
                
                if penalized_score > 0.1:
                    selected_sentences.append(sentences[idx])
                    selected_indices.append(idx)
        
        # Sort by original position
        selected_sentences = [x for _, x in sorted(zip(selected_indices, selected_sentences))]
        
        # Join and clean
        summary = ' '.join(selected_sentences)
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        return summary

    def summarize_pdf(self, pdf_path, num_sentences=5):
        """Main summarization function with comprehensive error handling."""
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            # Validate text
            if not text.strip():
                return {
                    "success": False,
                    "error": "No text could be extracted from the PDF"
                }
            
            # Generate summary
            summary = self.get_summary(text, num_sentences)
            
            # Validate summary
            if not summary.strip():
                return {
                    "success": False,
                    "error": "Could not generate a meaningful summary"
                }
            
            # Calculate statistics
            original_words = len(text.split())
            summary_words = len(summary.split())
            
            return {
                "success": True,
                "summary": summary,
                "original_length": original_words,
                "summary_length": summary_words
            }
            
        except ValueError as ve:
            return {
                "success": False,
                "error": str(ve)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"An error occurred: {str(e)}"
            }