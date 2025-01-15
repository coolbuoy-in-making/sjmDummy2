import { useState } from 'react';
import Hero from '../components/Home/Hero';
import TrustedBy from '../components/Home/TrustedBy';
import Categories from '../components/Home/Categories';
import AIButton from '../components/AIAssistant/AIButton';
import AIChatSidebar from '../components/AIAssistant/AIChatSidebar';

const Home = () => {
  const [isChatOpen, setIsChatOpen] = useState(false);

  return (
    <div>
      <Hero />
      <TrustedBy />
      <Categories />
      <AIButton onClick={() => setIsChatOpen(true)} />
      <AIChatSidebar 
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
      />
    </div>
  );
};

export default Home;