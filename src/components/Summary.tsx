import React from 'react';
import { BookOpen } from 'lucide-react';

interface SummaryProps {
  text: string;
}

const Summary: React.FC<SummaryProps> = ({ text }) => {
  return (
    <div className="mt-6">
      <div className="bg-indigo-50 rounded-lg p-6">
        <div className="flex items-center mb-4">
          <BookOpen className="w-6 h-6 text-indigo-600 mr-2" />
          <h3 className="text-xl font-semibold text-gray-900">Summary</h3>
        </div>
        <p className="text-gray-700 leading-relaxed whitespace-pre-line">
          {text}
        </p>
      </div>
    </div>
  );
};

export default Summary;

// import React from 'react';
// import { BookOpen, FileText, ArrowUpDown } from 'lucide-react';

// interface SummaryProps {
//   text: string;
//   originalLength?: number;
//   summaryLength?: number;
// }

// const Summary: React.FC<SummaryProps> = ({ text, originalLength, summaryLength }) => {
//   const compressionRatio = originalLength && summaryLength
//     ? ((originalLength - summaryLength) / originalLength * 100).toFixed(1)
//     : null;

//   return (
//     <div className="mt-6">
//       <div className="bg-indigo-50 rounded-lg p-6">
//         <div className="flex items-center mb-4">
//           <BookOpen className="w-6 h-6 text-indigo-600 mr-2" />
//           <h3 className="text-xl font-semibold text-gray-900">Summary</h3>
//         </div>
        
//         {compressionRatio && (
//           <div className="flex items-center gap-6 mb-4 p-4 bg-white rounded-lg">
//             <div className="flex items-center">
//               <FileText className="w-4 h-4 text-gray-500 mr-2" />
//               <span className="text-sm text-gray-600">
//                 Original: {originalLength} words
//               </span>
//             </div>
//             <div className="flex items-center">
//               <ArrowUpDown className="w-4 h-4 text-gray-500 mr-2" />
//               <span className="text-sm text-gray-600">
//                 Compressed: {compressionRatio}%
//               </span>
//             </div>
//             <div className="flex items-center">
//               <BookOpen className="w-4 h-4 text-gray-500 mr-2" />
//               <span className="text-sm text-gray-600">
//                 Summary: {summaryLength} words
//               </span>
//             </div>
//           </div>
//         )}
        
//         <p className="text-gray-700 leading-relaxed whitespace-pre-line">
//           {text}
//         </p>
//       </div>
//     </div>
//   );
// };

// export default Summary;


// import React from 'react';
// import { BookOpen, FileText, ArrowUpDown } from 'lucide-react';

// interface SummaryProps {
//   text: string;
//   originalLength?: number;
//   summaryLength?: number;
// }

// const Summary: React.FC<SummaryProps> = ({ text, originalLength, summaryLength }) => {
//   const compressionRatio = originalLength && summaryLength
//     ? ((originalLength - summaryLength) / originalLength * 100).toFixed(1)
//     : null;

//   return (
//     <div className="mt-6">
//       <div className="bg-indigo-50 rounded-lg p-6">
//         <div className="flex items-center mb-4">
//           <BookOpen className="w-6 h-6 text-indigo-600 mr-2" />
//           <h3 className="text-xl font-semibold text-gray-900">Summary</h3>
//         </div>
        
//         {compressionRatio && (
//           <div className="flex items-center gap-6 mb-4 p-4 bg-white rounded-lg">
//             <Stat icon={<FileText className="w-4 h-4 text-gray-500" />} label="Original" value={`${originalLength} words`} />
//             <Stat icon={<ArrowUpDown className="w-4 h-4 text-gray-500" />} label="Compressed" value={`${compressionRatio}%`} />
//             <Stat icon={<BookOpen className="w-4 h-4 text-gray-500" />} label="Summary" value={`${summaryLength} words`} />
//           </div>
//         )}
        
//         <p className="text-gray-700 leading-relaxed whitespace-pre-line">
//           {text}
//         </p>
//       </div>
//     </div>
//   );
// };

// const Stat = ({ icon, label, value }: { icon: JSX.Element; label: string; value: string }) => (
//   <div className="flex items-center">
//     {icon}
//     <span className="text-sm text-gray-600 ml-2">
//       {label}: {value}
//     </span>
//   </div>
// );

// Summary.defaultProps = {
//   originalLength: 0,
//   summaryLength: 0,
// };

// export default Summary;

// import React from 'react';
// import { BookOpen, FileText, ArrowUpDown } from 'lucide-react';

// interface SummaryProps {
//   text: string;
//   originalLength?: number;
//   summaryLength?: number;
// }

// const Summary: React.FC<SummaryProps> = ({ text, originalLength, summaryLength }) => {
//   const compressionRatio = originalLength && summaryLength
//     ? ((originalLength - summaryLength) / originalLength * 100).toFixed(1)
//     : null;

//   return (
//     <div className="mt-6">
//       <div className="bg-indigo-50 rounded-lg p-6">
//         <div className="flex items-center mb-4">
//           <BookOpen className="w-6 h-6 text-indigo-600 mr-2" />
//           <h3 className="text-xl font-semibold text-gray-900">Summary</h3>
//         </div>
        
//         {compressionRatio && (
//           <div className="flex items-center gap-6 mb-4 p-4 bg-white rounded-lg">
//             <div className="flex items-center">
//               <FileText className="w-4 h-4 text-gray-500 mr-2" />
//               <span className="text-sm text-gray-600">
//                 Original: {originalLength} words
//               </span>
//             </div>
//             <div className="flex items-center">
//               <ArrowUpDown className="w-4 h-4 text-gray-500 mr-2" />
//               <span className="text-sm text-gray-600">
//                 Compressed: {compressionRatio}%
//               </span>
//             </div>
//             <div className="flex items-center">
//               <BookOpen className="w-4 h-4 text-gray-500 mr-2" />
//               <span className="text-sm text-gray-600">
//                 Summary: {summaryLength} words
//               </span>
//             </div>
//           </div>
//         )}
        
//         <p className="text-gray-700 leading-relaxed whitespace-pre-line">
//           {text}
//         </p>
//       </div>
//     </div>
//   );
// };

// export default Summary;

