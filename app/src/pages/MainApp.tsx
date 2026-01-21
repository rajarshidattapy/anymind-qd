import { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import Navbar from '../components/Navbar';
import AgentsView from './AgentsView';
import MarketplaceView from './MarketplaceView';
import WalletView from './WalletView';
import Settings from './Settings';
import CapsuleDetail from './CapsuleDetail';
import { useApiClient } from '../lib/api';
import { useWallet } from '../contexts/WalletContextProvider';

interface LLMConfig {
  id: string;
  name: string;
  displayName: string;
  platform: string;
  apiKeyConfigured: boolean;
}

// Default agent IDs that should not be loaded as custom LLMs
const DEFAULT_AGENT_IDS: string[] = [];

const MainApp = () => {
  const location = useLocation();
  const [activeTab, setActiveTab] = useState('agents');
  const [activeSubTab, setActiveSubTab] = useState('');
  const [customLLMs, setCustomLLMs] = useState<LLMConfig[]>([]);
  const api = useApiClient();
  const { address, connected } = useWallet();
  const preferencesLoadedRef = useRef(false);

  // Check if we're on a capsule detail route
  const capsuleDetailMatch = location.pathname.match(/^\/app\/marketplace\/(.+)$/);
  const isCapsuleDetail = capsuleDetailMatch !== null;

  // If user navigates to /marketplace, set active tab to marketplace
  useEffect(() => {
    if (location.pathname === '/marketplace') {
      setActiveTab('marketplace');
    }
  }, [location.pathname]);

  // Load preferences from backend (Qdrant) on mount (only once)
  useEffect(() => {
    if (preferencesLoadedRef.current) return; // Only load once
    if (!connected || !address) return; // Need wallet to load preferences
    
    // Don't load preferences if we're on /marketplace route (URL takes priority)
    if (location.pathname === '/marketplace') {
      preferencesLoadedRef.current = true;
      return;
    }
    
    const loadPreferences = async () => {
      try {
        const prefs = await api.getPreferences();
        if (prefs.active_tab) {
          setActiveTab(prefs.active_tab);
        }
        if (prefs.active_sub_tab) {
          setActiveSubTab(prefs.active_sub_tab);
        }
        preferencesLoadedRef.current = true;
      } catch (error) {
        console.error('Error loading preferences:', error);
        preferencesLoadedRef.current = true; // Mark as loaded even on error to prevent retries
      }
    };

    loadPreferences();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected, address, location.pathname]); // api is stable, doesn't need to be in deps

  // Save preferences to backend (Qdrant) when they change
  useEffect(() => {
    const savePreferences = async () => {
      if (!connected || !address) return; // Need wallet to save preferences
      
      try {
        await api.updatePreferences({
          active_tab: activeTab,
          active_sub_tab: activeSubTab
        });
      } catch (error) {
        console.error('Error saving preferences:', error);
        // Fail silently - preferences are not critical
      }
    };

    // Debounce saves to avoid too many API calls
    const timeoutId = setTimeout(savePreferences, 500);
    return () => clearTimeout(timeoutId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, activeSubTab, connected, address]); // api is stable, doesn't need to be in deps

  // Load custom agents from backend when wallet is connected
  useEffect(() => {
    const loadCustomAgents = async () => {
      // Only load if wallet is connected (needed for filtering user's agents)
      if (!connected || !address) {
        setCustomLLMs([]);
        return;
      }
      
      try {
        const agents = await api.getAgents() as any[];
        
        // Filter out default agents and any GPT/Mistral agents, and convert to LLMConfig format
        // Allow specific exception: 'openai/gpt-oss-120b:free'
        const ALLOWED_MODELS = ['openai/gpt-oss-120b:free'];
        
        const isGPTOrMistral = (str: string): boolean => {
          const lower = str.toLowerCase();
          // Allow specific models
          if (ALLOWED_MODELS.some(allowed => lower.includes(allowed.toLowerCase()))) {
            return false;
          }
          // Check for exact matches or common patterns (but not substrings like "devstral")
          return lower === 'gpt' || lower === 'mistral' ||
                 /^gpt[-_\s]/.test(lower) || /[-_\s]gpt[-_\s]/.test(lower) || /[-_\s]gpt$/.test(lower) ||
                 /^mistral[-_\s]/.test(lower) || /[-_\s]mistral[-_\s]/.test(lower) || /[-_\s]mistral$/.test(lower);
        };
        
        const customAgents: LLMConfig[] = agents
          .filter(agent => {
            const id = (agent.id || '').toLowerCase();
            const name = (agent.name || '').toLowerCase();
            const displayName = ((agent.display_name || agent.name) || '').toLowerCase();
            // Check if this is an allowed model
            const isAllowed = ALLOWED_MODELS.some(allowed => 
              id.includes(allowed.toLowerCase()) || 
              name.includes(allowed.toLowerCase()) || 
              displayName.includes(allowed.toLowerCase())
            );
            if (isAllowed) return true;
            // Filter out default agent IDs and any GPT/Mistral references
            return !DEFAULT_AGENT_IDS.includes(agent.id) &&
                   !isGPTOrMistral(id) && !isGPTOrMistral(name) && !isGPTOrMistral(displayName);
          })
          .map(agent => ({
            id: agent.id,
            name: agent.name,
            displayName: agent.display_name || agent.name,
            platform: agent.platform,
            apiKeyConfigured: agent.api_key_configured || false
          }));
        
        setCustomLLMs(customAgents);
        console.log(`Loaded ${customAgents.length} custom agents from backend`);
      } catch (error) {
        console.error('Error loading custom agents:', error);
        // Continue with empty list if API fails
        setCustomLLMs([]);
      }
    };

    loadCustomAgents();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected, address]); // Reload when wallet connects/disconnects (api is stable)

  const handleAddLLM = (llm: LLMConfig) => {
    setCustomLLMs(prev => {
      const newLLMs = [...prev, llm];
      // If this is the first LLM added, automatically select it
      if (prev.length === 0 && newLLMs.length === 1) {
        setActiveSubTab(llm.id);
      }
      return newLLMs;
    });
  };

  const handleRemoveLLM = (llmId: string) => {
    setCustomLLMs(prev => {
      const filtered = prev.filter(llm => llm.id !== llmId);
      // If the deleted agent was selected, switch to first available or clear selection
      if (activeSubTab === llmId && filtered.length > 0) {
        setActiveSubTab(filtered[0].id);
      } else if (filtered.length === 0) {
        setActiveSubTab('');
      }
      return filtered;
    });
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'agents':
        return (
          <AgentsView 
            activeModel={activeSubTab} 
            customLLMs={customLLMs}
            onAddLLM={handleAddLLM}
            onRemoveLLM={handleRemoveLLM}
          />
        );
      case 'marketplace':
        return <MarketplaceView activeSubTab={activeSubTab} />;
      case 'wallet':
        return <WalletView activeSubTab={activeSubTab} />;
      case 'settings':
        return <Settings />;
      default:
        return (
          <AgentsView 
            activeModel={activeSubTab || ''} 
            customLLMs={customLLMs}
            onAddLLM={handleAddLLM}
          />
        );
    }
  };

  // If we're on a capsule detail page, render it without the tab navigation
  if (isCapsuleDetail) {
    return (
      <div className="min-h-screen bg-black">
        <CapsuleDetail />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        activeSubTab={activeSubTab}
        setActiveSubTab={setActiveSubTab}
        customLLMs={customLLMs}
        onAddLLM={handleAddLLM}
      />
      <main>
        {renderContent()}
      </main>
    </div>
  );
};

export default MainApp;
