import { useState } from 'react';
import Navigation from './components/Navigation';
import HomePage from './components/HomePage';
import AnalyzerPage from './components/AnalyzerPage';
import ResearchPage from './components/ResearchPage';
import DatasetPage from './components/DatasetPage';
import TeamPage from './components/TeamPage';

function App() {
  const [currentPage, setCurrentPage] = useState('home');

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage />;
      case 'analyzer':
        return <AnalyzerPage />;
      case 'research':
        return <ResearchPage />;
      case 'dataset':
        return <DatasetPage />;
      case 'team':
        return <TeamPage />;
      default:
        return <HomePage />;
    }
  };

  return (
    <div className="min-h-screen bg-[#0D1117]">
      <Navigation currentPage={currentPage} onNavigate={setCurrentPage} />
      {renderPage()}
    </div>
  );
}

export default App;
