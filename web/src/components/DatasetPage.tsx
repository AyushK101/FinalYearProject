import { Database, BarChart3, Layers, Download, CheckCircle } from 'lucide-react';

export default function DatasetPage() {
  const datasets = [
    {
      name: 'FaceForensics++',
      samples: '119,154',
      type: 'Video',
      methods: ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
      status: 'Primary'
    },
    {
      name: 'Celeb-DF v2',
      samples: '5,639',
      type: 'Video',
      methods: ['Celebrity Deepfakes'],
      status: 'Validation'
    },
    {
      name: 'DFDC (Preview)',
      samples: '23,654',
      type: 'Video',
      methods: ['Multiple GAN-based'],
      status: 'Testing'
    },
    {
      name: 'Custom Dataset',
      samples: '102,443',
      type: 'Mixed',
      methods: ['Proprietary Manipulation'],
      status: 'Augmentation'
    }
  ];

  const statistics = [
    { label: 'Total Videos', value: '250,890', change: '+12.5%', color: 'blue' },
    { label: 'Unique Subjects', value: '8,724', change: '+8.3%', color: 'purple' },
    { label: 'Manipulation Types', value: '12', change: '+2', color: 'green' },
    { label: 'Training Hours', value: '1,847', change: '+156h', color: 'orange' }
  ];

  return (
    <div className="min-h-screen bg-[#0D1117] text-white pt-24 pb-20">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f2937_1px,transparent_1px),linear-gradient(to_bottom,#1f2937_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)] opacity-20" />

      <div className="relative max-w-7xl mx-auto px-6">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">Dataset Overview</h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Comprehensive collection of authentic and manipulated media for training robust deepfake detection models
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
          {statistics.map((stat, idx) => (
            <div
              key={idx}
              className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-6 hover:border-blue-500/30 transition-all duration-300"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">{stat.label}</span>
                <span className={`text-xs text-${stat.color}-400 bg-${stat.color}-500/20 px-2 py-1 rounded`}>
                  {stat.change}
                </span>
              </div>
              <div className="text-3xl font-bold text-white">{stat.value}</div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {datasets.map((dataset, idx) => (
            <div
              key={idx}
              className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-6 hover:border-blue-500/30 transition-all duration-300"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <Database className="w-6 h-6 text-blue-400" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">{dataset.name}</h3>
                    <p className="text-sm text-gray-400">{dataset.type}</p>
                  </div>
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  dataset.status === 'Primary' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' :
                  dataset.status === 'Validation' ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' :
                  dataset.status === 'Testing' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
                  'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                }`}>
                  {dataset.status}
                </span>
              </div>

              <div className="mb-4">
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-gray-400">Samples</span>
                  <span className="text-white font-medium">{dataset.samples}</span>
                </div>
              </div>

              <div className="mb-4">
                <div className="text-sm text-gray-400 mb-2">Manipulation Methods:</div>
                <div className="flex flex-wrap gap-2">
                  {dataset.methods.map((method, midx) => (
                    <span
                      key={midx}
                      className="px-2 py-1 bg-gray-800/50 border border-gray-700 rounded text-xs text-gray-300"
                    >
                      {method}
                    </span>
                  ))}
                </div>
              </div>

              <button className="w-full px-4 py-2 bg-blue-500/20 border border-blue-500/30 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors text-sm font-medium">
                View Details
              </button>
            </div>
          ))}
        </div>

        <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8 mb-8">
          <h2 className="text-2xl font-bold mb-6">Data Distribution</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-4">By Manipulation Type</h3>
              <div className="space-y-3">
                {[
                  { label: 'Face Swap', value: 42 },
                  { label: 'Face Reenactment', value: 28 },
                  { label: 'Expression Transfer', value: 15 },
                  { label: 'Audio Sync', value: 10 },
                  { label: 'Other', value: 5 }
                ].map((item, idx) => (
                  <div key={idx}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-300">{item.label}</span>
                      <span className="text-blue-400 font-medium">{item.value}%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                        style={{ width: `${item.value}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-4">By Data Source</h3>
              <div className="space-y-3">
                {[
                  { label: 'Public Datasets', value: 65 },
                  { label: 'Custom Collection', value: 25 },
                  { label: 'Synthetic Generation', value: 10 }
                ].map((item, idx) => (
                  <div key={idx}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-300">{item.label}</span>
                      <span className="text-purple-400 font-medium">{item.value}%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                        style={{ width: `${item.value}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6">Data Processing Pipeline</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { step: 'Collection', desc: 'Aggregate from multiple sources', icon: Database },
              { step: 'Preprocessing', desc: 'Face detection & alignment', icon: Layers },
              { step: 'Quality Control', desc: 'Filtering & validation', icon: CheckCircle },
              { step: 'Augmentation', desc: 'Data balancing & expansion', icon: BarChart3 }
            ].map((item, idx) => {
              const Icon = item.icon;
              return (
                <div
                  key={idx}
                  className="bg-gray-900/50 border border-white/5 rounded-lg p-6 text-center hover:border-blue-500/30 transition-colors"
                >
                  <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <Icon className="w-6 h-6 text-blue-400" />
                  </div>
                  <div className="text-sm font-semibold text-white mb-1">{item.step}</div>
                  <div className="text-xs text-gray-400">{item.desc}</div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="mt-8 bg-gradient-to-br from-blue-500/10 to-purple-500/10 backdrop-blur-xl border border-blue-500/20 rounded-xl p-8">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xl font-semibold mb-2">Access Dataset Documentation</h3>
              <p className="text-gray-400">
                Download comprehensive dataset specifications, preprocessing scripts, and usage guidelines.
              </p>
            </div>
            <button className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-medium hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-200 flex items-center space-x-2">
              <Download className="w-5 h-5" />
              <span>Download</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
