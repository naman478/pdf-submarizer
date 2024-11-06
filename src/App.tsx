import { useState } from 'react';
import { FileText, Upload, BookOpen, Loader2 } from 'lucide-react';
import { Document, Page, pdfjs } from 'react-pdf';
import PDFUploader from './components/PDFUploader';
import Summary from './components/Summary';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [numPages, setNumPages] = useState<number>(0);
  const [summary, setSummary] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (file: File) => {
    setFile(file);
    setSummary('');
  };

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  };

  const generateSummary = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setSummary(data.summary);
      } else {
        setSummary(`Error: ${data.error}`);
      }
    } catch (error) {
      setSummary('Error generating summary');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <FileText className="w-12 h-12 text-indigo-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">PDF Summarizer</h1>
          <p className="text-lg text-gray-600">
            Upload your PDF document and get an AI-powered summary
          </p>
        </header>

        <div className="max-w-4xl mx-auto">
          {!file ? (
            <PDFUploader onFileChange={handleFileChange} />
          ) : (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center">
                  <BookOpen className="w-6 h-6 text-indigo-600 mr-2" />
                  <h2 className="text-xl font-semibold">{file.name}</h2>
                </div>
                <button
                  onClick={() => setFile(null)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  Change file
                </button>
              </div>

              <div className="mb-6">
                <Document
                  file={file}
                  onLoadSuccess={onDocumentLoadSuccess}
                  className="border rounded-lg overflow-hidden"
                >
                  <Page
                    pageNumber={1}
                    width={window.innerWidth > 768 ? 600 : 300}
                    renderTextLayer={true}
                    renderAnnotationLayer={true}
                  />
                </Document>
                <p className="text-sm text-gray-500 mt-2">
                  Page 1 of {numPages}
                </p>
              </div>

              {!summary && (
                <button
                  onClick={generateSummary}
                  disabled={loading}
                  className="w-full flex items-center justify-center px-4 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:bg-indigo-400"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Generating Summary...
                    </>
                  ) : (
                    <>
                      <Upload className="w-5 h-5 mr-2" />
                      Generate Summary
                    </>
                  )}
                </button>
              )}

              {summary && <Summary text={summary} />}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;


// Set up PDF.js worker
// import { useState } from 'react';
// import { FileText, Upload, BookOpen, Loader2 } from 'lucide-react';
// import { Document, Page, pdfjs } from 'react-pdf';
// import PDFUploader from './components/PDFUploader';
// import Summary from './components/Summary';
// import 'react-pdf/dist/Page/AnnotationLayer.css';
// import 'react-pdf/dist/Page/TextLayer.css';

// // Set up PDF.js worker
// pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

// interface SummaryResponse {
//   success: boolean;
//   summary: string;
//   original_length?: number;
//   summary_length?: number;
//   error?: string;
// }

// function App() {
//   const [file, setFile] = useState<File | null>(null);
//   const [numPages, setNumPages] = useState<number>(0);
//   const [summary, setSummary] = useState<string>('');
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState<string>('');
//   const [stats, setStats] = useState<{
//     original_length?: number;
//     summary_length?: number;
//   }>({});

//   const handleFileChange = (file: File) => {
//     setFile(file);
//     setSummary('');
//     setError('');
//     setStats({});
//   };

//   const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
//     setNumPages(numPages);
//   };

//   const generateSummary = async () => {
//     if (!file) {
//       setError("No file uploaded. Please upload a PDF file.");
//       return;
//     }

//     setLoading(true);
//     setError('');
//     setSummary('');
//     setStats({});

//     const formData = new FormData();
//     formData.append('file', file);

//     try {
//       const response = await fetch('http://localhost:5000/upload', {
//         method: 'POST',
//         body: formData,
//       });

//       if (!response.ok) {
//         throw new Error('Network response was not ok');
//       }

//       const data: SummaryResponse = await response.json();

//       if (data.success) {
//         setSummary(data.summary);
//         setStats({
//           original_length: data.original_length,
//           summary_length: data.summary_length,
//         });
//       } else {
//         setError(data.error || 'Failed to generate summary');
//       }
//     } catch (err) {
//       setError(`Connection error: Please ensure the backend is running.`);
//       console.error('Error:', err);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
//       <div className="container mx-auto px-4 py-8">
//         <header className="text-center mb-12">
//           <div className="flex items-center justify-center mb-4">
//             <FileText className="w-12 h-12 text-indigo-600" />
//           </div>
//           <h1 className="text-4xl font-bold text-gray-900 mb-2">Enhanced PDF Summarizer</h1>
//           <p className="text-lg text-gray-600">
//             Upload your PDF document and get an AI-powered summary using TextRank algorithm
//           </p>
//         </header>

//         <div className="max-w-4xl mx-auto">
//           {!file ? (
//             <PDFUploader onFileChange={handleFileChange} />
//           ) : (
//             <div className="bg-white rounded-xl shadow-lg p-6">
//               <div className="flex items-center justify-between mb-6">
//                 <div className="flex items-center">
//                   <BookOpen className="w-6 h-6 text-indigo-600 mr-2" />
//                   <h2 className="text-xl font-semibold">{file.name}</h2>
//                 </div>
//                 <button
//                   onClick={() => setFile(null)}
//                   className="text-gray-500 hover:text-gray-700"
//                 >
//                   Change file
//                 </button>
//               </div>

//               <div className="mb-6">
//                 <Document
//                   file={file}
//                   onLoadSuccess={onDocumentLoadSuccess}
//                   className="border rounded-lg overflow-hidden"
//                 >
//                   <Page
//                     pageNumber={1}
//                     width={window.innerWidth > 768 ? 600 : 300}
//                     renderTextLayer={true}
//                     renderAnnotationLayer={true}
//                   />
//                 </Document>
//                 <p className="text-sm text-gray-500 mt-2">
//                   Page 1 of {numPages}
//                 </p>
//               </div>

//               {error && (
//                 <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-lg">
//                   {error}
//                 </div>
//               )}

//               {!summary && !error && (
//                 <button
//                   onClick={generateSummary}
//                   disabled={loading}
//                   className="w-full flex items-center justify-center px-4 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:bg-indigo-400"
//                 >
//                   {loading ? (
//                     <>
//                       <Loader2 className="w-5 h-5 mr-2 animate-spin" />
//                       Generating Summary...
//                     </>
//                   ) : (
//                     <>
//                       <Upload className="w-5 h-5 mr-2" />
//                       Generate Summary
//                     </>
//                   )}
//                 </button>
//               )}

//               {summary && (
//                 <Summary 
//                   text={summary}
//                   originalLength={stats.original_length}
//                   summaryLength={stats.summary_length}
//                 />
//               )}
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }

// export default App;

