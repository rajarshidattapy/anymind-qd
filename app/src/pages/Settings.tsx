import { useState, useEffect } from 'react';
import { Settings as SettingsIcon } from 'lucide-react';
import { useApiClient } from '../lib/api';
import { useWallet } from '../contexts/WalletContextProvider';

const Settings = () => {
  const [defaultModel, setDefaultModel] = useState('default');
  const [memoryBehavior, setMemoryBehavior] = useState('adaptive');
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const api = useApiClient();
  const { address, connected } = useWallet();

  // Load preferences from backend (Qdrant) on mount
  useEffect(() => {
    const loadPreferences = async () => {
      if (!connected || !address) return;
      
      try {
        const prefs = await api.getPreferences();
        if (prefs.default_model) {
          setDefaultModel(prefs.default_model);
        }
        if (prefs.memory_behavior) {
          setMemoryBehavior(prefs.memory_behavior);
        }
      } catch (error) {
        console.error('Error loading preferences:', error);
        // Continue with defaults if API fails
      }
    };

    loadPreferences();
  }, [connected, address, api]);

  const handleSave = async () => {
    if (!connected || !address) {
      alert('Please connect your wallet to save preferences');
      return;
    }

    setIsSaving(true);
    setSaveStatus('idle');
    
    try {
      await api.updatePreferences({
        default_model: defaultModel,
        memory_behavior: memoryBehavior
      });
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 2000);
    } catch (error) {
      console.error('Error saving preferences:', error);
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 3000);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Settings</h1>
          <p className="text-gray-400">Customize your MantleMind experience</p>
        </div>

        <div className="space-y-8">
          {/* AI Model Settings */}
          <div className="bg-gray-800 rounded-xl border border-gray-700 p-6">
            <div className="flex items-center space-x-2 mb-6">
              <SettingsIcon className="h-5 w-5 text-blue-400" />
              <h2 className="text-xl font-semibold text-white">AI Model Settings</h2>
            </div>
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-3">
                  Default Model
                </label>
                <select
                  value={defaultModel}
                  onChange={(e) => setDefaultModel(e.target.value)}
                  className="w-full bg-gray-700 text-white px-4 py-3 rounded-lg border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                >
                  <option value="default">Default Model</option>
                </select>
                <p className="text-xs text-gray-500 mt-2">
                  Choose your preferred AI model for new conversations
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-3">
                  Memory Behavior
                </label>
                <select
                  value={memoryBehavior}
                  onChange={(e) => setMemoryBehavior(e.target.value)}
                  className="w-full bg-gray-700 text-white px-4 py-3 rounded-lg border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                >
                  <option value="adaptive">Adaptive - AI decides when to remember</option>
                  <option value="aggressive">Aggressive - Remember everything</option>
                  <option value="conservative">Conservative - Only remember important items</option>
                  <option value="manual">Manual - User controls memory</option>
                </select>
                <p className="text-xs text-gray-500 mt-2">
                  Controls how actively the system builds memory from conversations
                </p>
              </div>
            </div>
          </div>
          {/* Save Button */}
          <div className="flex justify-end items-center space-x-4">
            {saveStatus === 'success' && (
              <span className="text-green-400 text-sm">✓ Saved successfully!</span>
            )}
            {saveStatus === 'error' && (
              <span className="text-red-400 text-sm">✗ Failed to save</span>
            )}
            {!connected && (
              <span className="text-yellow-400 text-sm">Connect wallet to save preferences</span>
            )}
            <button 
              onClick={handleSave}
              disabled={isSaving || !connected}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg font-semibold transition-colors"
            >
              {isSaving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
