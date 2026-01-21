import { useWallet } from '../contexts/WalletContextProvider';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
if (!API_BASE_URL) {
  throw new Error('VITE_API_BASE_URL environment variable is required');
}

// Ensure the URL has a protocol
const normalizeUrl = (url: string): string => {
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return `http://${url}`;
  }
  return url;
};

export class ApiClient {
  private baseUrl: string;
  private getWalletAddress: () => string | null;

  constructor(getWalletAddress: () => string | null) {
    this.baseUrl = normalizeUrl(API_BASE_URL);
    this.getWalletAddress = getWalletAddress;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const walletAddress = this.getWalletAddress();
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (walletAddress) {
      (headers as Record<string, string>)['X-Wallet-Address'] = walletAddress;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorMessage = 'Unknown error';
      try {
        const error = await response.json();
        errorMessage = error.message || error.detail || `HTTP ${response.status}`;
      } catch {
        if (response.status === 0) {
          errorMessage = 'Cannot connect to server. Check your internet connection and API URL.';
        } else {
          errorMessage = `HTTP error! status: ${response.status}`;
        }
      }
      throw new Error(errorMessage);
    }

    return response.json();
  }

  // Agents
  async getAgents() {
    return this.request('/api/v1/agents/');
  }

  async createAgent(agent: {
    name: string;
    display_name: string;
    platform: string;
    api_key: string;
    model?: string;
  }) {
    return this.request('/api/v1/agents/', {
      method: 'POST',
      body: JSON.stringify(agent),
    });
  }

  async stakeOnAgent(agentId: string, data: {
    stake_amount: number;
    price_per_query: number;
    category: string;
    description: string;
    payment_signature?: string;
  }) {
    return this.request(`/api/v1/agents/${encodeURIComponent(agentId)}/stake`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Chats
  async getChats(agentId: string) {
    return this.request(`/api/v1/agents/${encodeURIComponent(agentId)}/chats`);
  }

  async createChat(agentId: string, chat: { name: string; memory_size?: string; web_search_enabled?: boolean }) {
    return this.request(`/api/v1/agents/${encodeURIComponent(agentId)}/chats`, {
      method: 'POST',
      body: JSON.stringify({
        ...chat,
        agent_id: agentId, // Required by backend schema
      }),
    });
  }

  async getChat(agentId: string, chatId: string) {
    return this.request(`/api/v1/agents/${encodeURIComponent(agentId)}/chats/${encodeURIComponent(chatId)}`);
  }

  async updateAgent(agentId: string, update: { display_name?: string; model?: string }) {
    return this.request(`/api/v1/agents/${encodeURIComponent(agentId)}`, {
      method: 'PUT',
      body: JSON.stringify(update),
    });
  }

  async updateChat(agentId: string, chatId: string, update: { name?: string; memory_size?: string; web_search_enabled?: boolean }) {
    return this.request(`/api/v1/agents/${encodeURIComponent(agentId)}/chats/${encodeURIComponent(chatId)}`, {
      method: 'PUT',
      body: JSON.stringify(update),
    });
  }

  async sendMessage(agentId: string, chatId: string, message: { role: string; content: string }) {
    return this.request(`/api/v1/agents/${encodeURIComponent(agentId)}/chats/${encodeURIComponent(chatId)}/messages`, {
      method: 'POST',
      body: JSON.stringify(message),
    });
  }

  async sendMessageStream(
    agentId: string,
    chatId: string,
    message: { role: string; content: string },
    onChunk: (chunk: string) => void,
    onComplete: () => void,
    onError: (error: Error) => void
  ) {
    const walletAddress = this.getWalletAddress();
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (walletAddress) {
      headers['X-Wallet-Address'] = walletAddress;
    }

    const response = await fetch(
      `${this.baseUrl}/api/v1/agents/${encodeURIComponent(agentId)}/chats/${encodeURIComponent(chatId)}/messages/stream`,
      {
        method: 'POST',
        headers,
        body: JSON.stringify(message),
      }
    );

    if (!response.ok) {
      let errorMessage = 'Unknown error';
      try {
        const error = await response.json();
        errorMessage = error.message || error.detail || `HTTP ${response.status}`;
      } catch {
        if (response.status === 0) {
          errorMessage = 'Cannot connect to server. Check your internet connection and API URL.';
        } else {
          errorMessage = `HTTP error! status: ${response.status}`;
        }
      }
      throw new Error(errorMessage);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('No response body');
    }

    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            try {
              const data = JSON.parse(dataStr);
              if (data.done) {
                onComplete();
                return;
              }
              if (data.content) {
                onChunk(data.content);
              }
              if (data.error) {
                onError(new Error(data.error));
                return;
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }

      // Process remaining buffer
      if (buffer.startsWith('data: ')) {
        const dataStr = buffer.slice(6);
        try {
          const data = JSON.parse(dataStr);
          if (data.content) {
            onChunk(data.content);
          }
        } catch (e) {
          // Skip invalid JSON
        }
      }

      onComplete();
    } catch (error) {
      onError(error instanceof Error ? error : new Error('Stream error'));
    }
  }

  async getMessages(agentId: string, chatId: string) {
    return this.request(`/api/v1/agents/${encodeURIComponent(agentId)}/chats/${encodeURIComponent(chatId)}/messages`);
  }

  async deleteChat(agentId: string, chatId: string) {
    return this.request(`/api/v1/agents/${encodeURIComponent(agentId)}/chats/${encodeURIComponent(chatId)}`, {
      method: 'DELETE',
    });
  }

  async deleteAgent(agentId: string) {
    return this.request(`/api/v1/agents/${encodeURIComponent(agentId)}`, {
      method: 'DELETE',
    });
  }

  // Capsules
  async getCapsules() {
    return this.request('/api/v1/capsules/');
  }

  async createCapsule(capsule: {
    name: string;
    description: string;
    category: string;
    price_per_query: number;
    metadata?: Record<string, any>;
  }) {
    return this.request('/api/v1/capsules/', {
      method: 'POST',
      body: JSON.stringify(capsule),
    });
  }

  async getCapsule(capsuleId: string) {
    return this.request(`/api/v1/capsules/${encodeURIComponent(capsuleId)}`);
  }

  async queryCapsule(capsuleId: string, payload: {
    prompt: string;
    payment_signature?: string;
    amount_paid?: number;
  }) {
    return this.request(`/api/v1/capsules/${encodeURIComponent(capsuleId)}/query`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // Marketplace
  async browseMarketplace(filters?: {
    category?: string;
    min_reputation?: number;
    max_price?: number;
    sort_by?: string;
    limit?: number;
    offset?: number;
  }) {
    const params = new URLSearchParams();
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          params.append(key, value.toString());
        }
      });
    }
    const query = params.toString();
    return this.request(`/api/v1/marketplace${query ? `?${query}` : ''}`);
  }

  async getTrending(limit = 10) {
    return this.request(`/api/v1/marketplace/trending?limit=${limit}`);
  }

  async searchCapsules(query: string, limit = 20) {
    return this.request(`/api/v1/marketplace/search?q=${encodeURIComponent(query)}&limit=${limit}`);
  }

  // Wallet
  async getBalance() {
    return this.request('/api/v1/wallet/balance');
  }

  async getEarnings(period?: string) {
    const query = period ? `?period=${period}` : '';
    return this.request(`/api/v1/wallet/earnings${query}`);
  }

  async getStakingInfo() {
    return this.request('/api/v1/wallet/staking');
  }

  async createStaking(staking: { capsule_id: string; stake_amount: number }) {
    return this.request('/api/v1/wallet/staking', {
      method: 'POST',
      body: JSON.stringify(staking),
    });
  }

  // Preferences (stored in backend persistence layer)
  async getPreferences() {
    return this.request<{
      default_model?: string;
      memory_behavior?: string;
      active_tab?: string;
      active_sub_tab?: string;
    }>('/api/v1/preferences/');
  }

  async updatePreferences(preferences: {
    default_model?: string;
    memory_behavior?: string;
    active_tab?: string;
    active_sub_tab?: string;
  }) {
    return this.request('/api/v1/preferences/', {
      method: 'POST',
      body: JSON.stringify(preferences),
    });
  }

  async clearPreferences() {
    return this.request('/api/v1/preferences/', {
      method: 'DELETE',
    });
  }
}

// Hook to use API client
export function useApiClient() {
  const { address } = useWallet();
  
  const getWalletAddress = () => {
    return address || null;
  };

  return new ApiClient(getWalletAddress);
}

