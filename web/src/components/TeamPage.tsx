import { Github, Linkedin, Mail, Award } from 'lucide-react';

export default function TeamPage() {
  const team = [
    {
      name: 'Dr. Sarah Chen',
      role: 'Research Lead & Principal Investigator',
      expertise: 'Deep Learning, Computer Vision',
      bio: 'PhD in Computer Vision from MIT. 10+ years experience in adversarial learning and synthetic media detection.',
      achievements: ['Published 45+ papers', 'ACM Fellow', 'Best Paper Award CVPR 2023']
    },
    {
      name: 'Marcus Rodriguez',
      role: 'Model Architecture Engineer',
      expertise: 'Neural Networks, CNN/LSTM Architectures',
      bio: 'Specialized in temporal modeling and attention mechanisms. Previously at DeepMind working on video understanding systems.',
      achievements: ['15+ publications', 'Google Research Award', 'ICCV Outstanding Reviewer']
    },
    {
      name: 'Dr. Aisha Patel',
      role: 'Dataset & Evaluation Engineer',
      expertise: 'Data Engineering, Evaluation Metrics',
      bio: 'Expert in large-scale dataset curation and bias mitigation. Led data initiatives at Stanford AI Lab.',
      achievements: ['Created FaceDB-V2 dataset', 'NeurIPS Dataset Track', '20K+ citations']
    },
    {
      name: 'Liam O\'Connor',
      role: 'Backend & Deployment Lead',
      expertise: 'MLOps, System Architecture',
      bio: 'Full-stack engineer with focus on scalable ML inference systems. Built real-time detection pipelines at Meta.',
      achievements: ['5+ production systems', 'Open-source contributor', 'AWS ML Hero']
    }
  ];

  return (
    <div className="min-h-screen bg-[#0D1117] text-white pt-24 pb-20">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f2937_1px,transparent_1px),linear-gradient(to_bottom,#1f2937_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)] opacity-20" />

      <div className="relative max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <div className="inline-block px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-full text-blue-400 text-sm font-medium mb-4">
            Meet Our Team
          </div>
          <h1 className="text-4xl font-bold mb-4">Research Contributors</h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            A multidisciplinary team of researchers, engineers, and domain experts dedicated to advancing media forensics technology.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
          {team.map((member, idx) => (
            <div
              key={idx}
              className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8 hover:border-blue-500/30 transition-all duration-300 group"
            >
              <div className="flex items-start space-x-6">
                <div className="w-24 h-24 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center flex-shrink-0 text-3xl font-bold">
                  {member.name.split(' ').map(n => n[0]).join('')}
                </div>

                <div className="flex-1">
                  <h3 className="text-xl font-bold mb-1">{member.name}</h3>
                  <div className="text-blue-400 text-sm font-medium mb-2">{member.role}</div>
                  <div className="text-gray-400 text-sm mb-4">
                    <span className="font-medium">Expertise:</span> {member.expertise}
                  </div>

                  <div className="flex space-x-2 mb-4">
                    <button className="p-2 bg-white/5 border border-white/10 rounded-lg hover:bg-blue-500/20 hover:border-blue-500/30 transition-colors">
                      <Linkedin className="w-4 h-4 text-gray-400 group-hover:text-blue-400" />
                    </button>
                    <button className="p-2 bg-white/5 border border-white/10 rounded-lg hover:bg-blue-500/20 hover:border-blue-500/30 transition-colors">
                      <Github className="w-4 h-4 text-gray-400 group-hover:text-blue-400" />
                    </button>
                    <button className="p-2 bg-white/5 border border-white/10 rounded-lg hover:bg-blue-500/20 hover:border-blue-500/30 transition-colors">
                      <Mail className="w-4 h-4 text-gray-400 group-hover:text-blue-400" />
                    </button>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-800">
                <p className="text-sm text-gray-400 leading-relaxed mb-4">{member.bio}</p>

                <div className="space-y-2">
                  {member.achievements.map((achievement, aidx) => (
                    <div key={aidx} className="flex items-center space-x-2 text-xs">
                      <Award className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                      <span className="text-gray-400">{achievement}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl border border-white/10 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6 text-center">Collaborations & Affiliations</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              'MIT Computer Science',
              'Stanford AI Lab',
              'Carnegie Mellon University',
              'Meta AI Research',
              'Google Research',
              'OpenAI',
              'Berkeley BAIR',
              'Microsoft Research'
            ].map((org, idx) => (
              <div
                key={idx}
                className="bg-gray-900/50 border border-white/5 rounded-lg p-4 text-center hover:border-blue-500/30 transition-colors"
              >
                <div className="text-sm text-gray-300 font-medium">{org}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-12 bg-gradient-to-br from-blue-500/10 to-purple-500/10 backdrop-blur-xl border border-blue-500/20 rounded-xl p-8 text-center">
          <h3 className="text-xl font-semibold mb-3">Join Our Research</h3>
          <p className="text-gray-400 mb-6 max-w-2xl mx-auto">
            We're always looking for talented researchers and engineers to contribute to cutting-edge media forensics technology.
          </p>
          <button className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-medium hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-200">
            View Open Positions
          </button>
        </div>
      </div>
    </div>
  );
}
