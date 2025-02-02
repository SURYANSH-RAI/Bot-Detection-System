import React, { useState } from 'react';
import { Upload, AlertTriangle, FileText, Cpu } from 'lucide-react';
import Papa from 'papaparse';

// Simple Alert Component to replace shadcn/ui Alert
const Alert = ({ children, variant }) => (
  <div className={`p-4 rounded-lg ${
    variant === 'destructive'
      ? 'bg-red-900/50 border-red-800 text-red-200'
      : 'bg-gray-800/50 border-gray-700 text-gray-200'
  } border backdrop-blur-sm`}>
    {children}
  </div>
);

const CircuitBackground = () => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none opacity-10">
    <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
      <pattern id="circuit" x="0" y="0" width="50" height="50" patternUnits="userSpaceOnUse">
        <path d="M0 25h50 M25 0v50" stroke="currentColor" strokeWidth="0.5" fill="none"/>
        <circle cx="25" cy="25" r="3" fill="currentColor"/>
        <circle cx="0" cy="25" r="3" fill="currentColor"/>
        <circle cx="50" cy="25" r="3" fill="currentColor"/>
        <circle cx="25" cy="0" r="3" fill="currentColor"/>
        <circle cx="25" cy="50" r="3" fill="currentColor"/>
      </pattern>
      <rect width="100%" height="100%" fill="url(#circuit)"/>
    </svg>
  </div>
);

const GlowingOrb = () => (
  <div className="absolute top-10 right-10 w-32 h-32 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 blur-2xl opacity-20 animate-pulse"/>
);

const BotDetectionSystem = () => {
  const [files, setFiles] = useState([]);
  const [socialLinks, setSocialLinks] = useState('');
  const [profileLinks, setProfileLinks] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [analysisMethod, setAnalysisMethod] = useState('ml');

  const analysisOptions = [
    {
      id: 'ml',
      name: 'Machine Learning',
      description: 'Uses trained neural network for advanced pattern detection'
    },
    {
      id: 'rule-based',
      name: 'Rule-Based',
      description: 'Uses heuristic rules and statistical analysis'
    }
  ];

  const handleFileUpload = (event) => {
    const uploadedFiles = Array.from(event.target.files);
    setFiles(uploadedFiles);
  };

  const handleSocialLinksChange = (event) => {
    setSocialLinks(event.target.value);
  };

  const handleProfileLinksChange = (event) => {
    setProfileLinks(event.target.value);
  };

  const processData = async () => {
    setLoading(true);
    setError('');
   
    try {
      // Parse CSV files
      const fileResults = await Promise.all(
        files.map((file) => {
          return new Promise((resolve) => {
            Papa.parse(file, {
              complete: (results) => resolve(results.data),
              header: true,
              skipEmptyLines: true
            });
          });
        })
      );

      // Here you would normally send the data to your backend
      // For now, using mock data
      // Prepare data to send
      const analysisData = {
        csvData: fileResults,
        socialLinks: socialLinks.split('\n').filter(link => link.trim()),
        profileLinks: profileLinks.split('\n').filter(link => link.trim())
      };

      // Call appropriate API endpoint based on analysis method
      const response = await fetch(`http://localhost:8000/api/${analysisMethod === 'ml' ? 'ml-analysis' : 'rule-based-analysis'}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Access-Control-Allow-Origin': '*'
        },
        credentials: 'include',
        body: JSON.stringify(analysisData)
      });

      if (!response.ok) {
        throw new Error('Failed to get analysis results');
      }

      const analysisResults = await response.json();

      setResults(analysisResults);
    } catch (err) {
      setError('Error processing data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 relative overflow-hidden">
      <CircuitBackground />
      <GlowingOrb />
     
      <div className="max-w-4xl mx-auto p-6 space-y-6 relative">
        <div className="flex items-center gap-3 mb-8">
          <Cpu className="w-8 h-8 text-blue-400" />
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Bot Detection System
          </h1>
        </div>

        <div className="space-y-4 relative z-10">
          <div className="bg-gray-800/50 rounded-lg p-4 backdrop-blur-sm border border-gray-700">
            <h3 className="text-blue-400 font-medium mb-3">Select Analysis Method</h3>
            <div className="grid grid-cols-2 gap-4">
              {analysisOptions.map((option) => (
                <button
                  key={option.id}
                  onClick={() => setAnalysisMethod(option.id)}
                  className={`p-4 rounded-lg border transition-all duration-200 text-left space-y-2 ${
                    analysisMethod === option.id
                      ? 'border-blue-500 bg-blue-500/10'
                      : 'border-gray-700 bg-gray-900/50 hover:border-gray-600'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <div className={`w-4 h-4 rounded-full border-2 ${
                      analysisMethod === option.id
                        ? 'border-blue-500 bg-blue-500'
                        : 'border-gray-600'
                    }`} />
                    <span className="font-medium text-gray-200">{option.name}</span>
                  </div>
                  <p className="text-sm text-gray-400">{option.description}</p>
                </button>
              ))}
            </div>
          </div>

          <div className="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center bg-gray-800/50 backdrop-blur-sm">
            <input
              type="file"
              multiple
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer flex flex-col items-center group"
            >
              <div className="p-4 rounded-full bg-gray-700/50 group-hover:bg-gray-600/50 transition-colors">
                <Upload className="w-8 h-8 text-blue-400" />
              </div>
              <span className="text-sm text-gray-400 mt-2">
                Upload CSV files with social media data
              </span>
            </label>
            {files.length > 0 && (
              <div className="mt-4">
                <h3 className="font-medium text-blue-400">Uploaded Files:</h3>
                {files.map((file, index) => (
                  <div key={index} className="flex items-center gap-2 text-sm text-gray-300">
                    <FileText className="w-4 h-4" />
                    {file.name}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-300">
              Social Media Post Links
            </label>
            <textarea
              value={socialLinks}
              onChange={handleSocialLinksChange}
              className="w-full p-2 bg-gray-800/50 border border-gray-700 rounded-md text-gray-200 backdrop-blur-sm"
              placeholder="Enter social media post links (one per line)"
              rows="3"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-300">
              Profile Links
            </label>
            <textarea
              value={profileLinks}
              onChange={handleProfileLinksChange}
              className="w-full p-2 bg-gray-800/50 border border-gray-700 rounded-md text-gray-200 backdrop-blur-sm"
              placeholder="Enter profile links (one per line)"
              rows="3"
            />
          </div>

          <button
            onClick={processData}
            disabled={loading}
            className="w-full py-2 px-4 rounded-md bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 backdrop-blur-sm"
          >
            {loading ? 'Processing...' : 'Analyze Data'}
          </button>
        </div>

        {error && (
          <Alert variant="destructive">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              <p className="font-medium">Error</p>
            </div>
            <p className="mt-2">{error}</p>
          </Alert>
        )}

        {results && (
          <div className="mt-6 p-6 bg-gray-800/50 rounded-lg backdrop-blur-sm border border-gray-700">
            <h2 className="text-xl font-semibold mb-4 text-blue-400">Analysis Results</h2>
           
            <div className="space-y-4">
              <div className="p-4 bg-gray-900/50 rounded-lg border border-gray-700">
                <h3 className="font-medium mb-2 text-gray-300">Bot Probability Score</h3>
                <div className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                  {(results.botProbability * 100).toFixed(1)}%
                </div>
                {analysisMethod === 'ml' && (
                  <div className="mt-2 text-sm text-gray-400">
                    ML Confidence: {(results.mlConfidence * 100).toFixed(1)}%
                  </div>
                )}
                {analysisMethod === 'rule-based' && (
                  <div className="mt-2 text-sm text-gray-400">
                    Matched Rules: {results.ruleMatches} of 5
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gray-900/50 rounded-lg border border-gray-700">
                  <h3 className="font-medium mb-2 text-gray-300">Content Analysis</h3>
                  <div className="space-y-2">
                    <div>
                      <span className="text-gray-400">Spam Score:</span>
                      <span className="ml-2 font-medium text-blue-400">
                        {(results.contentPatterns.spamScore * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Sentiment Score:</span>
                      <span className="ml-2 font-medium text-blue-400">
                        {(results.contentPatterns.sentimentScore * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-gray-900/50 rounded-lg border border-gray-700">
                  <h3 className="font-medium mb-2 text-gray-300">Behavioral Metrics</h3>
                  <div className="space-y-2">
                    <div>
                      <span className="text-gray-400">Posting Frequency:</span>
                      <span className="ml-2 font-medium text-blue-400">
                        {results.behavioralMetrics.postingFrequency} posts/day
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Engagement Rate:</span>
                      <span className="ml-2 font-medium text-blue-400">
                        {(results.behavioralMetrics.engagementRate * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BotDetectionSystem;
