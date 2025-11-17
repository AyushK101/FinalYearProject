import { Brain, Database, Layers, TrendingUp, CheckCircle2, FileText } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#0D1117] text-white">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f2937_1px,transparent_1px),linear-gradient(to_bottom,#1f2937_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)] opacity-20" />

      <div className="relative max-w-7xl mx-auto px-6 pt-32 pb-20">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center mb-24">
          <div>
            <div className="inline-block px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-full text-blue-400 text-sm font-medium mb-6">
              Research Project v1.3.2
            </div>
            <h1 className="text-5xl font-bold mb-6 leading-tight bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              Deepfake Detection & Media Forensics Research
            </h1>
            <p className="text-xl text-gray-400 mb-8 leading-relaxed">
              A Machine Learning Approach for Identifying Manipulated Videos and Images
            </p>
            <div className="flex space-x-4">
              <button className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-medium hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-200">
                Try Live Demo
              </button>
              <button className="px-6 py-3 bg-white/5 border border-white/10 rounded-lg font-medium hover:bg-white/10 transition-all duration-200">
                Read Paper
              </button>
            </div>
          </div>

          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20 rounded-2xl blur-3xl" />
            <div className="relative bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl border border-white/10 rounded-2xl p-8">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                  <Brain className="w-8 h-8 text-blue-400 mb-2" />
                  <div className="text-2xl font-bold text-white">98.7%</div>
                  <div className="text-xs text-gray-400">Detection Accuracy</div>
                </div>
                <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                  <Database className="w-8 h-8 text-purple-400 mb-2" />
                  <div className="text-2xl font-bold text-white">250K+</div>
                  <div className="text-xs text-gray-400">Training Samples</div>
                </div>
                <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                  <Layers className="w-8 h-8 text-green-400 mb-2" />
                  <div className="text-2xl font-bold text-white">4.2M</div>
                  <div className="text-xs text-gray-400">Model Parameters</div>
                </div>
                <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-4">
                  <TrendingUp className="w-8 h-8 text-orange-400 mb-2" />
                  <div className="text-2xl font-bold text-white">0.97</div>
                  <div className="text-xs text-gray-400">AUC-ROC Score</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-24">
          <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-6 hover:border-blue-500/30 transition-all duration-300">
            <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
              <FileText className="w-6 h-6 text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Abstract</h3>
            <p className="text-sm text-gray-400 leading-relaxed">
              We present a deep learning framework for detecting deepfake videos using spatiotemporal feature analysis.
              Our hybrid CNN-LSTM architecture achieves state-of-the-art performance across multiple benchmark datasets.
            </p>
          </div>

          <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-6 hover:border-purple-500/30 transition-all duration-300">
            <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4">
              <Brain className="w-6 h-6 text-purple-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Motivation</h3>
            <p className="text-sm text-gray-400 leading-relaxed">
              The proliferation of synthetic media poses significant threats to information integrity.
              Our research addresses this challenge through advanced computer vision and temporal modeling techniques.
            </p>
          </div>

          <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-6 hover:border-green-500/30 transition-all duration-300">
            <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
              <Database className="w-6 h-6 text-green-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Dataset</h3>
            <p className="text-sm text-gray-400 leading-relaxed">
              Trained on FaceForensics++, Celeb-DF, and proprietary datasets comprising 250,000+ videos
              with diverse manipulation techniques and demographic representations.
            </p>
          </div>
        </div>

        <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8 mb-24">
          <h2 className="text-2xl font-bold mb-6">Research Objectives</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              'Develop robust detection algorithms for various deepfake generation methods',
              'Create interpretable models with attention-based visualization capabilities',
              'Achieve real-time inference performance for practical deployment',
              'Build transfer learning approaches for low-resource scenarios',
              'Establish comprehensive evaluation protocols and benchmarks',
              'Integrate multi-modal analysis for audio-visual consistency checking'
            ].map((objective, idx) => (
              <div key={idx} className="flex items-start space-x-3">
                <CheckCircle2 className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-gray-300">{objective}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8 mb-24">
          <h2 className="text-2xl font-bold mb-6">Methodology Pipeline</h2>
          <div className="flex items-center justify-between overflow-x-auto pb-4">
            {[
              { label: 'Input Video', icon: FileText, color: 'blue' },
              { label: 'Preprocessing', icon: Layers, color: 'purple' },
              { label: 'Frame Extraction', icon: Database, color: 'green' },
              { label: 'CNN Features', icon: Brain, color: 'orange' },
              { label: 'Temporal Model', icon: TrendingUp, color: 'pink' },
              { label: 'Classification', icon: CheckCircle2, color: 'cyan' }
            ].map((step, idx) => {
              const Icon = step.icon;
              return (
                <div key={idx} className="flex items-center">
                  <div className="flex flex-col items-center min-w-[120px]">
                    <div className={`w-16 h-16 bg-${step.color}-500/20 border border-${step.color}-500/30 rounded-xl flex items-center justify-center mb-2`}>
                      <Icon className={`w-8 h-8 text-${step.color}-400`} />
                    </div>
                    <p className="text-xs text-gray-400 text-center">{step.label}</p>
                  </div>
                  {idx < 5 && (
                    <div className="w-12 h-0.5 bg-gradient-to-r from-blue-500/50 to-purple-500/50 mx-2" />
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6">Model Evaluation Results</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gray-900/50 rounded-lg p-6 border border-white/5">
              <h3 className="text-sm font-medium text-gray-400 mb-4">Accuracy Metrics</h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">Accuracy</span>
                    <span className="text-blue-400 font-medium">98.7%</span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-blue-500 to-blue-400" style={{ width: '98.7%' }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">Precision</span>
                    <span className="text-purple-400 font-medium">97.2%</span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-purple-500 to-purple-400" style={{ width: '97.2%' }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">Recall</span>
                    <span className="text-green-400 font-medium">96.8%</span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-green-500 to-green-400" style={{ width: '96.8%' }} />
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-gray-900/50 rounded-lg p-6 border border-white/5">
              <h3 className="text-sm font-medium text-gray-400 mb-4">Confusion Matrix</h3>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-green-500/20 border border-green-500/30 rounded p-3 text-center">
                  <div className="text-2xl font-bold text-green-400">24,356</div>
                  <div className="text-xs text-gray-400 mt-1">True Negative</div>
                </div>
                <div className="bg-red-500/20 border border-red-500/30 rounded p-3 text-center">
                  <div className="text-2xl font-bold text-red-400">412</div>
                  <div className="text-xs text-gray-400 mt-1">False Positive</div>
                </div>
                <div className="bg-red-500/20 border border-red-500/30 rounded p-3 text-center">
                  <div className="text-2xl font-bold text-red-400">538</div>
                  <div className="text-xs text-gray-400 mt-1">False Negative</div>
                </div>
                <div className="bg-green-500/20 border border-green-500/30 rounded p-3 text-center">
                  <div className="text-2xl font-bold text-green-400">23,894</div>
                  <div className="text-xs text-gray-400 mt-1">True Positive</div>
                </div>
              </div>
            </div>

            <div className="bg-gray-900/50 rounded-lg p-6 border border-white/5">
              <h3 className="text-sm font-medium text-gray-400 mb-4">ROC Curve Analysis</h3>
              <div className="space-y-4">
                <div>
                  <div className="text-3xl font-bold text-blue-400">0.974</div>
                  <div className="text-xs text-gray-400">AUC-ROC Score</div>
                </div>
                <div className="h-32 bg-gray-800/50 rounded flex items-end justify-center p-2">
                  <div className="text-xs text-gray-500 text-center">ROC visualization placeholder</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-12 bg-gradient-to-br from-blue-500/10 to-purple-500/10 backdrop-blur-xl border border-blue-500/20 rounded-xl p-6">
          <h3 className="text-sm font-semibold text-blue-400 mb-2">How to Cite This Project</h3>
          <div className="bg-gray-900/50 rounded-lg p-4 font-mono text-xs text-gray-300">
            @article&#123;deepguard2024,<br />
            &nbsp;&nbsp;title=&#123;Deepfake Detection and Media Forensics: A Deep Learning Approach&#125;,<br />
            &nbsp;&nbsp;author=&#123;Research Team&#125;,<br />
            &nbsp;&nbsp;journal=&#123;Computer Vision Research&#125;,<br />
            &nbsp;&nbsp;year=&#123;2024&#125;<br />
            &#125;
          </div>
        </div>
      </div>
    </div>
  );
}
