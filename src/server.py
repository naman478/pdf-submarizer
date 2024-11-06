from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pdf_summarizer import PDFSummarizer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

summarizer = PDFSummarizer()

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"})

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"success": False, "error": "Only PDF files are supported"})

    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Call the PDF summarizer to summarize the PDF
        summary = summarizer.summarize_pdf(file_path)

        # Clean up the uploaded file
        os.remove(file_path)
        
        return jsonify({
            "success": True,
            "summary": summary
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error processing PDF: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True)
