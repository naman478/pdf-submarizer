import React, { useCallback } from 'react';
import { Upload } from 'lucide-react';

interface PDFUploaderProps {
  onFileChange: (file: File) => void;
}

const PDFUploader: React.FC<PDFUploaderProps> = ({ onFileChange }) => {
  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type === 'application/pdf') {
        onFileChange(file);
      }
    },
    [onFileChange]
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      onFileChange(file);
    }
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      className="border-2 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-indigo-500 transition-colors"
    >
      <div className="flex flex-col items-center">
        <Upload className="w-12 h-12 text-gray-400 mb-4" />
        <h3 className="text-xl font-semibold text-gray-700 mb-2">
          Drop your PDF here
        </h3>
        <p className="text-gray-500 mb-6">or</p>
        <label className="cursor-pointer bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors">
          Browse Files
          <input
            type="file"
            className="hidden"
            accept="application/pdf"
            onChange={handleFileInput}
          />
        </label>
        <p className="text-sm text-gray-500 mt-4">
          Supports PDF files (2-5 pages)
        </p>
      </div>
    </div>
  );
};

export default PDFUploader;