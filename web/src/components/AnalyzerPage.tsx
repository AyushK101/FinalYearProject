import { useState } from 'react';
import { Upload, PlayCircle, Download, FileVideo, AlertTriangle, CheckCircle, Activity, Clock, Film, Monitor } from 'lucide-react';

export default function AnalyzerPage() {
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const handleFileUpload = () => {
    setUploadedFile('sample_video.mp4');
    setTimeout(() => {
      setIsProcessing(true);
      setCurrentStep(0);

      const interval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= 5) {
            clearInterval(interval);
            setIsProcessing(false);
            setShowResults(true);
            return prev;
          }
          return prev + 1;
        });
      }, 800);
    }, 500);
  };

  const processingSteps = [
    { label: 'Receiving Input', icon: Upload, color: 'blue' },
    { label: 'Preprocessing', icon: Activity, color: 'purple' },
    { label: 'Frame Extraction', icon: Film, color: 'green' },
    { label: 'CNN Analysis', icon: Monitor, color: 'orange' },
    { label: 'Temporal Modeling', icon: Clock, color: 'pink' },
    { label: 'Final Classification', icon: CheckCircle, color: 'cyan' }
  ];

  const frameAnalysis = [
    { frame: 1, score: 0.12, status: 'real' },
    { frame: 24, score: 0.18, status: 'real' },
    { frame: 48, score: 0.23, status: 'real' },
    { frame: 72, score: 0.89, status: 'fake' },
    { frame: 96, score: 0.92, status: 'fake' },
    { frame: 120, score: 0.87, status: 'fake' },
    { frame: 144, score: 0.15, status: 'real' },
    { frame: 168, score: 0.21, status: 'real' },
  ];

  return (
    <div className="min-h-screen bg-[#0D1117] text-white pt-24 pb-20">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f2937_1px,transparent_1px),linear-gradient(to_bottom,#1f2937_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)] opacity-20" />

      <div className="relative max-w-7xl mx-auto px-6">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">Deepfake Analyzer</h1>
          <p className="text-gray-400">Upload a video or image for AI-powered authenticity verification</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2">
            {!uploadedFile ? (
              <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border-2 border-dashed border-gray-700 hover:border-blue-500/50 rounded-xl p-12 text-center transition-all duration-300 cursor-pointer"
                onClick={handleFileUpload}>
                <div className="w-20 h-20 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Upload className="w-10 h-10 text-blue-400" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Drop your media file here</h3>
                <p className="text-gray-400 mb-6">or click to browse (MP4, AVI, MOV, JPG, PNG)</p>
                <button className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-medium hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-200">
                  Select File
                </button>
              </div>
            ) : (
              <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden">
                <div className="bg-gray-900/50 aspect-video flex items-center justify-center">
                  <div className="text-center">
                    <FileVideo className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                    <p className="text-gray-500">Video Preview</p>
                  </div>
                </div>
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <FileVideo className="w-5 h-5 text-blue-400" />
                      <span className="font-medium">{uploadedFile}</span>
                    </div>
                    <button className="p-2 hover:bg-white/5 rounded-lg transition-colors">
                      <PlayCircle className="w-5 h-5 text-gray-400" />
                    </button>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Duration:</span>
                      <span className="ml-2 text-white">00:12</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Frames:</span>
                      <span className="ml-2 text-white">360</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Resolution:</span>
                      <span className="ml-2 text-white">1920x1080</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">File Metadata</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between py-2 border-b border-gray-800">
                <span className="text-gray-400">Format</span>
                <span className="text-white">{uploadedFile ? 'MP4' : '-'}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-800">
                <span className="text-gray-400">Codec</span>
                <span className="text-white">{uploadedFile ? 'H.264' : '-'}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-800">
                <span className="text-gray-400">Bitrate</span>
                <span className="text-white">{uploadedFile ? '5.2 Mbps' : '-'}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-800">
                <span className="text-gray-400">FPS</span>
                <span className="text-white">{uploadedFile ? '30' : '-'}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-800">
                <span className="text-gray-400">Size</span>
                <span className="text-white">{uploadedFile ? '7.8 MB' : '-'}</span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-gray-400">Audio</span>
                <span className="text-white">{uploadedFile ? 'AAC, 128kbps' : '-'}</span>
              </div>
            </div>
          </div>
        </div>

        {isProcessing && (
          <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8 mb-8">
            <h3 className="text-xl font-semibold mb-6 text-center">Processing Pipeline</h3>
            <div className="flex items-center justify-between mb-8">
              {processingSteps.map((step, idx) => {
                const Icon = step.icon;
                const isActive = idx === currentStep;
                const isCompleted = idx < currentStep;

                return (
                  <div key={idx} className="flex items-center">
                    <div className="flex flex-col items-center min-w-[100px]">
                      <div className={`w-14 h-14 rounded-xl flex items-center justify-center mb-2 transition-all duration-300 ${
                        isActive
                          ? `bg-${step.color}-500/30 border-2 border-${step.color}-500 shadow-lg shadow-${step.color}-500/50`
                          : isCompleted
                          ? `bg-${step.color}-500/20 border border-${step.color}-500/50`
                          : 'bg-gray-800/50 border border-gray-700'
                      }`}>
                        <Icon className={`w-7 h-7 ${isActive || isCompleted ? `text-${step.color}-400` : 'text-gray-600'}`} />
                      </div>
                      <p className={`text-xs text-center ${isActive ? 'text-white font-medium' : 'text-gray-500'}`}>
                        {step.label}
                      </p>
                    </div>
                    {idx < processingSteps.length - 1 && (
                      <div className={`w-8 h-0.5 mb-6 transition-all duration-300 ${
                        isCompleted ? 'bg-blue-500' : 'bg-gray-800'
                      }`} />
                    )}
                  </div>
                );
              })}
            </div>
            <div className="bg-gray-900/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Analysis Progress</span>
                <span className="text-sm text-blue-400 font-medium">{Math.round((currentStep / 5) * 100)}%</span>
              </div>
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                  style={{ width: `${(currentStep / 5) * 100}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {showResults && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
              <div className="bg-gradient-to-br from-red-500/10 to-red-600/10 backdrop-blur-xl border border-red-500/30 rounded-xl p-8 text-center">
                <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-4" />
                <div className="text-5xl font-bold text-red-400 mb-2">87.3%</div>
                <div className="text-gray-300 font-medium mb-1">Deepfake Probability</div>
                <div className="text-sm text-gray-500">High confidence detection</div>
              </div>

              <div className="lg:col-span-2 bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8">
                <h3 className="text-lg font-semibold mb-6">Classification Result</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <AlertTriangle className="w-4 h-4 text-red-400" />
                        <span className="text-sm font-medium text-gray-300">Fake</span>
                      </div>
                      <span className="text-red-400 font-bold">87.3%</span>
                    </div>
                    <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-red-500 to-red-400" style={{ width: '87.3%' }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <span className="text-sm font-medium text-gray-300">Real</span>
                      </div>
                      <span className="text-green-400 font-bold">12.7%</span>
                    </div>
                    <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-green-500 to-green-400" style={{ width: '12.7%' }} />
                    </div>
                  </div>
                </div>

                <div className="mt-6 flex space-x-3">
                  <button className="flex-1 px-4 py-2 bg-blue-500/20 border border-blue-500/30 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors">
                    <Download className="w-4 h-4 inline mr-2" />
                    Download Report
                  </button>
                  <button className="flex-1 px-4 py-2 bg-purple-500/20 border border-purple-500/30 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors">
                    View Frame Analysis
                  </button>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8 mb-8">
              <h3 className="text-lg font-semibold mb-6">Per-Frame Confidence Timeline</h3>
              <div className="h-48 bg-gray-900/50 rounded-lg p-4 flex items-end space-x-1">
                {Array.from({ length: 50 }).map((_, idx) => {
                  const height = Math.random() * 100;
                  const isSuspicious = height > 70;
                  return (
                    <div
                      key={idx}
                      className={`flex-1 rounded-t transition-all duration-200 hover:opacity-75 ${
                        isSuspicious ? 'bg-red-500/70' : 'bg-blue-500/50'
                      }`}
                      style={{ height: `${height}%` }}
                    />
                  );
                })}
              </div>
              <div className="flex items-center justify-between mt-4 text-sm text-gray-400">
                <span>Frame 0</span>
                <span>Deepfake confidence over time</span>
                <span>Frame 360</span>
              </div>
            </div>

            <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8">
              <h3 className="text-lg font-semibold mb-6">Suspicious Frames Detected</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {frameAnalysis.filter(f => f.status === 'fake').map((frame, idx) => (
                  <div key={idx} className="bg-gray-900/50 border border-red-500/30 rounded-lg p-3 hover:border-red-500/50 transition-colors cursor-pointer">
                    <div className="aspect-video bg-gray-800 rounded mb-3 flex items-center justify-center">
                      <Film className="w-8 h-8 text-gray-600" />
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-400">Frame {frame.frame}</span>
                      <span className="text-red-400 font-medium">{(frame.score * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
