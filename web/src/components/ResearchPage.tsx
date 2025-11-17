import { BookOpen, TrendingUp, Target, Zap, GitCompare } from 'lucide-react';

export default function ResearchPage() {
  const models = [
    {
      name: 'Xception',
      accuracy: 95.2,
      speed: 'Fast',
      params: '22.9M',
      description: 'Depthwise separable convolutions'
    },
    {
      name: 'ResNet-50',
      accuracy: 93.8,
      speed: 'Medium',
      params: '25.6M',
      description: 'Residual learning framework'
    },
    {
      name: 'EfficientNet-B4',
      accuracy: 96.1,
      speed: 'Medium',
      params: '19.3M',
      description: 'Compound scaling method'
    },
    {
      name: 'CNN-LSTM (Ours)',
      accuracy: 98.7,
      speed: 'Slow',
      params: '42.1M',
      description: 'Spatiotemporal hybrid architecture'
    }
  ];

  return (
    <div className="min-h-screen bg-[#0D1117] text-white pt-24 pb-20">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f2937_1px,transparent_1px),linear-gradient(to_bottom,#1f2937_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)] opacity-20" />

      <div className="relative max-w-7xl mx-auto px-6">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">Research Methodology</h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            In-depth analysis of our approach to deepfake detection using advanced deep learning techniques
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8">
            <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
              <BookOpen className="w-6 h-6 text-blue-400" />
            </div>
            <h2 className="text-2xl font-bold mb-4">Problem Statement</h2>
            <p className="text-gray-400 leading-relaxed mb-4">
              The rapid advancement of generative adversarial networks (GANs) and deepfake technology has made it
              increasingly difficult to distinguish authentic media from manipulated content. This poses significant
              risks to information integrity, privacy, and security.
            </p>
            <p className="text-gray-400 leading-relaxed">
              Our research addresses this challenge by developing a robust, interpretable detection system that
              combines spatial feature extraction with temporal consistency analysis.
            </p>
          </div>

          <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8">
            <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4">
              <Target className="w-6 h-6 text-purple-400" />
            </div>
            <h2 className="text-2xl font-bold mb-4">Our Approach</h2>
            <div className="space-y-3">
              {[
                'Extract per-frame spatial features using CNN encoder',
                'Model temporal dependencies with bidirectional LSTM',
                'Incorporate attention mechanisms for interpretability',
                'Apply multi-task learning for robustness',
                'Ensemble predictions across multiple timescales'
              ].map((step, idx) => (
                <div key={idx} className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-purple-500/20 rounded flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs text-purple-400 font-bold">{idx + 1}</span>
                  </div>
                  <p className="text-gray-300 text-sm">{step}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8 mb-12">
          <h2 className="text-2xl font-bold mb-6">Architecture Overview</h2>
          <div className="bg-gray-900/50 rounded-lg p-8 mb-6">
            <div className="text-center text-gray-500 py-12">
              <div className="text-sm mb-2">Model Architecture Diagram</div>
              <div className="text-xs">CNN Encoder → Temporal Modeling → Attention Layer → Classification Head</div>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-900/50 border border-white/5 rounded-lg p-4">
              <div className="text-sm font-medium text-blue-400 mb-2">Spatial Branch</div>
              <div className="text-xs text-gray-400">
                EfficientNet-B4 backbone pre-trained on ImageNet, fine-tuned for face manipulation artifacts
              </div>
            </div>
            <div className="bg-gray-900/50 border border-white/5 rounded-lg p-4">
              <div className="text-sm font-medium text-purple-400 mb-2">Temporal Branch</div>
              <div className="text-xs text-gray-400">
                Bidirectional LSTM with 512 hidden units captures frame-to-frame inconsistencies
              </div>
            </div>
            <div className="bg-gray-900/50 border border-white/5 rounded-lg p-4">
              <div className="text-sm font-medium text-green-400 mb-2">Fusion Module</div>
              <div className="text-xs text-gray-400">
                Multi-head attention aggregates spatial and temporal cues for final prediction
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8 mb-12">
          <div className="flex items-center space-x-3 mb-6">
            <GitCompare className="w-6 h-6 text-blue-400" />
            <h2 className="text-2xl font-bold">Model Comparison</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Model</th>
                  <th className="text-center py-3 px-4 text-sm font-medium text-gray-400">Accuracy</th>
                  <th className="text-center py-3 px-4 text-sm font-medium text-gray-400">Parameters</th>
                  <th className="text-center py-3 px-4 text-sm font-medium text-gray-400">Speed</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Description</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model, idx) => (
                  <tr key={idx} className="border-b border-gray-800/50 hover:bg-white/5 transition-colors">
                    <td className="py-4 px-4">
                      <div className="font-medium text-white">{model.name}</div>
                    </td>
                    <td className="py-4 px-4 text-center">
                      <div className="inline-flex items-center space-x-2">
                        <span className={`font-bold ${
                          model.accuracy > 97 ? 'text-green-400' :
                          model.accuracy > 95 ? 'text-blue-400' : 'text-gray-400'
                        }`}>
                          {model.accuracy}%
                        </span>
                      </div>
                    </td>
                    <td className="py-4 px-4 text-center text-gray-300">{model.params}</td>
                    <td className="py-4 px-4 text-center">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        model.speed === 'Fast' ? 'bg-green-500/20 text-green-400' :
                        model.speed === 'Medium' ? 'bg-blue-500/20 text-blue-400' :
                        'bg-orange-500/20 text-orange-400'
                      }`}>
                        {model.speed}
                      </span>
                    </td>
                    <td className="py-4 px-4 text-sm text-gray-400">{model.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
          <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8">
            <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
              <TrendingUp className="w-6 h-6 text-green-400" />
            </div>
            <h2 className="text-xl font-bold mb-4">Key Innovations</h2>
            <ul className="space-y-3">
              {[
                'Hybrid spatiotemporal architecture combining CNNs and LSTMs',
                'Attention-based frame weighting for interpretability',
                'Multi-scale temporal modeling (1, 4, 8, 16 frame windows)',
                'Transfer learning from face recognition pre-training',
                'Adversarial training for improved robustness'
              ].map((innovation, idx) => (
                <li key={idx} className="flex items-start space-x-2 text-sm text-gray-300">
                  <Zap className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>{innovation}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8">
            <h2 className="text-xl font-bold mb-6">Training Details</h2>
            <div className="space-y-4">
              <div className="flex justify-between py-3 border-b border-gray-800">
                <span className="text-gray-400">Optimizer</span>
                <span className="text-white font-medium">Adam (β₁=0.9, β₂=0.999)</span>
              </div>
              <div className="flex justify-between py-3 border-b border-gray-800">
                <span className="text-gray-400">Learning Rate</span>
                <span className="text-white font-medium">1e-4 (cosine decay)</span>
              </div>
              <div className="flex justify-between py-3 border-b border-gray-800">
                <span className="text-gray-400">Batch Size</span>
                <span className="text-white font-medium">32</span>
              </div>
              <div className="flex justify-between py-3 border-b border-gray-800">
                <span className="text-gray-400">Epochs</span>
                <span className="text-white font-medium">50</span>
              </div>
              <div className="flex justify-between py-3 border-b border-gray-800">
                <span className="text-gray-400">Loss Function</span>
                <span className="text-white font-medium">Binary Cross-Entropy</span>
              </div>
              <div className="flex justify-between py-3">
                <span className="text-gray-400">Hardware</span>
                <span className="text-white font-medium">8× NVIDIA A100 GPUs</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 backdrop-blur-xl border border-blue-500/20 rounded-xl p-8">
          <h3 className="text-xl font-semibold mb-3">Publications & Citations</h3>
          <p className="text-gray-400 mb-4">
            This research has been accepted at CVPR 2024 and is available on arXiv.
          </p>
          <div className="flex space-x-4">
            <button className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-medium hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-200">
              Read Full Paper
            </button>
            <button className="px-6 py-3 bg-white/5 border border-white/10 rounded-lg font-medium hover:bg-white/10 transition-all duration-200">
              View on arXiv
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
