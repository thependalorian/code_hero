# Frontend Integration Guide

## 🚀 Overview

This guide explains how to integrate the Code Hero Next.js frontend with the FastAPI backend. The frontend features a modern glassmorphism design with real-time chat, agent management, and project tracking capabilities.

## 🏗️ Architecture

### Frontend Stack
- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS + DaisyUI
- **Design**: Glassmorphism inspired by Imagica.ai
- **State Management**: React hooks (useState, useEffect, useRef)
- **HTTP Client**: Fetch API (can be extended with Axios)
- **Real-time**: WebSocket ready (future enhancement)

### Backend Integration
- **API Base URL**: `http://localhost:8000`
- **CORS**: Enabled for frontend communication
- **Authentication**: Ready for implementation
- **Error Handling**: Comprehensive error responses

## 📁 Component Structure

```
frontend/src/components/
├── agents/
│   ├── AgentCard.tsx          # Individual agent display
│   └── AgentGrid.tsx          # Agent grid layout
├── chat/
│   └── ChatInterface.tsx      # Real-time chat component
├── documents/
│   ├── DocumentCard.tsx       # Document display and actions
│   ├── DocumentGrid.tsx       # Document grid layout
│   └── DocumentList.tsx       # Document list layout
├── projects/
│   ├── ProjectCard.tsx        # Project display and tracking
│   └── ProjectGrid.tsx        # Project grid layout
└── ui/
    ├── Button.tsx             # Gradient and glass buttons
    ├── GlassCard.tsx          # Glassmorphism container
    ├── Input.tsx              # Glassmorphism input fields
    └── Modal.tsx              # Animated modal dialogs
```

## 🔌 API Integration

### 1. API Client Setup

Create an API client utility:

```typescript
// utils/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class ApiClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Chat API methods
  async sendMessage(message: string, conversationId?: string) {
    return this.request('/api/chat/', {
      method: 'POST',
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
      }),
    });
  }

  async getChatHistory(conversationId: string) {
    return this.request(`/api/chat/${conversationId}`);
  }

  // Search API methods
  async searchDocuments(query: string, collection: string = 'strategy_book', limit: number = 5) {
    return this.request('/api/astra/search', {
      method: 'POST',
      body: JSON.stringify({
        query,
        collection,
        limit,
      }),
    });
  }

  // Multi-agent coordination
  async coordinateTask(taskDescription: string, projectId?: string) {
    return this.request('/multi-agent/coordinate', {
      method: 'POST',
      body: JSON.stringify({
        task_description: taskDescription,
        project_id: projectId,
      }),
    });
  }

  // Health check
  async healthCheck() {
    return this.request('/health');
  }
}

export const apiClient = new ApiClient();
```

### 2. React Hooks for API Integration

Create custom hooks for common operations:

```typescript
// hooks/useChat.ts
import { useState, useCallback } from 'react';
import { apiClient } from '@/utils/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export function useChat(conversationId?: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(async (content: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.sendMessage(content, conversationId);
      setMessages(response.messages || []);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [conversationId]);

  const loadHistory = useCallback(async () => {
    if (!conversationId) return;

    setIsLoading(true);
    try {
      const response = await apiClient.getChatHistory(conversationId);
      setMessages(response.messages || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load chat history');
    } finally {
      setIsLoading(false);
    }
  }, [conversationId]);

  return {
    messages,
    sendMessage,
    loadHistory,
    isLoading,
    error,
  };
}
```

```typescript
// hooks/useAgents.ts
import { useState, useEffect } from 'react';

interface Agent {
  id: string;
  name: string;
  role: string;
  status: 'active' | 'processing' | 'idle' | 'error';
  capabilities: string[];
  performance: {
    tasksCompleted: number;
    successRate: number;
    avgResponseTime: string;
  };
}

export function useAgents() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate agent data - replace with actual API call
    const mockAgents: Agent[] = [
      {
        id: 'strategic_expert',
        name: 'Strategic Expert',
        role: 'Strategy & Planning',
        status: 'active',
        capabilities: ['Project Planning', 'Architecture Design', 'Risk Assessment'],
        performance: {
          tasksCompleted: 156,
          successRate: 94,
          avgResponseTime: '2.3s',
        },
      },
      {
        id: 'fastapi_expert',
        name: 'FastAPI Expert',
        role: 'Backend Development',
        status: 'processing',
        capabilities: ['API Design', 'Database Integration', 'Authentication'],
        performance: {
          tasksCompleted: 89,
          successRate: 97,
          avgResponseTime: '1.8s',
        },
      },
      // Add more agents...
    ];

    setTimeout(() => {
      setAgents(mockAgents);
      setIsLoading(false);
    }, 1000);
  }, []);

  return { agents, isLoading };
}
```

### 3. Environment Configuration

Set up environment variables:

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## 🎨 Component Integration Examples

### Chat Interface Integration

```typescript
// components/chat/ChatInterface.tsx
'use client';

import { useState, useRef, useEffect } from 'react';
import { useChat } from '@/hooks/useChat';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';

export function ChatInterface() {
  const [input, setInput] = useState('');
  const { messages, sendMessage, isLoading } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const message = input.trim();
    setInput('');

    try {
      await sendMessage(message);
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  return (
    <div className="flex flex-col h-full glass rounded-2xl p-6">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-2xl ${
                message.role === 'user'
                  ? 'bg-gradient-primary text-white rounded-br-md'
                  : 'glass rounded-bl-md'
              }`}
            >
              <p className="text-sm">{message.content}</p>
              <p className="text-xs opacity-70 mt-1">
                {new Date(message.timestamp).toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="glass rounded-2xl rounded-bl-md px-4 py-2">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-100"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-200"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me anything..."
          className="flex-1"
          disabled={isLoading}
        />
        <Button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="px-6"
        >
          Send
        </Button>
      </form>
    </div>
  );
}
```

### Agent Status Integration

```typescript
// components/agents/AgentCard.tsx
'use client';

import { GlassCard } from '@/components/ui/GlassCard';

interface Agent {
  id: string;
  name: string;
  role: string;
  status: 'active' | 'processing' | 'idle' | 'error';
  capabilities: string[];
  performance: {
    tasksCompleted: number;
    successRate: number;
    avgResponseTime: string;
  };
}

interface AgentCardProps {
  agent: Agent;
  onSelect?: (agent: Agent) => void;
}

export function AgentCard({ agent, onSelect }: AgentCardProps) {
  const statusColors = {
    active: 'bg-green-500',
    processing: 'bg-yellow-500',
    idle: 'bg-gray-500',
    error: 'bg-red-500',
  };

  return (
    <GlassCard className="p-6 hover:scale-105 transition-transform cursor-pointer">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{agent.name}</h3>
          <p className="text-sm text-gray-600">{agent.role}</p>
        </div>
        <div className="flex items-center space-x-2">
          <div
            className={`w-3 h-3 rounded-full ${statusColors[agent.status]}`}
          />
          <span className="text-xs capitalize text-gray-600">
            {agent.status}
          </span>
        </div>
      </div>

      {/* Capabilities */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Capabilities</h4>
        <div className="flex flex-wrap gap-1">
          {agent.capabilities.map((capability, index) => (
            <span
              key={index}
              className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full"
            >
              {capability}
            </span>
          ))}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Tasks Completed</span>
          <span className="font-medium">{agent.performance.tasksCompleted}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Success Rate</span>
          <span className="font-medium">{agent.performance.successRate}%</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Avg Response</span>
          <span className="font-medium">{agent.performance.avgResponseTime}</span>
        </div>
      </div>

      {onSelect && (
        <button
          onClick={() => onSelect(agent)}
          className="w-full mt-4 btn-gradient text-sm py-2 rounded-lg"
        >
          Select Agent
        </button>
      )}
    </GlassCard>
  );
}
```

## 🔄 Real-time Features

### WebSocket Integration (Future Enhancement)

```typescript
// hooks/useWebSocket.ts
import { useEffect, useRef, useState } from 'react';

export function useWebSocket(url: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<any[]>([]);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      setIsConnected(true);
    };

    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setMessages((prev) => [...prev, message]);
    };

    ws.current.onclose = () => {
      setIsConnected(false);
    };

    return () => {
      ws.current?.close();
    };
  }, [url]);

  const sendMessage = (message: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    }
  };

  return { isConnected, messages, sendMessage };
}
```

## 🎯 Page Integration

### Dashboard Page

```typescript
// app/page.tsx
'use client';

import { useEffect, useState } from 'react';
import { AgentCard } from '@/components/agents/AgentCard';
import { ProjectCard } from '@/components/projects/ProjectCard';
import { GlassCard } from '@/components/ui/GlassCard';
import { useAgents } from '@/hooks/useAgents';
import { apiClient } from '@/utils/api';

export default function Dashboard() {
  const { agents, isLoading: agentsLoading } = useAgents();
  const [systemHealth, setSystemHealth] = useState<any>(null);

  useEffect(() => {
    // Load system health
    apiClient.healthCheck()
      .then(setSystemHealth)
      .catch(console.error);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-mesh p-6">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-4xl font-bold gradient-text mb-4">
            Code Hero Dashboard
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Orchestrate your development workflow with AI-powered agents
          </p>
        </div>

        {/* System Status */}
        {systemHealth && (
          <GlassCard className="p-6">
            <h2 className="text-xl font-semibold mb-4">System Status</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {systemHealth.status === 'healthy' ? '✓' : '✗'}
                </div>
                <div className="text-sm text-gray-600">System Health</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {agents.length}
                </div>
                <div className="text-sm text-gray-600">Active Agents</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">24</div>
                <div className="text-sm text-gray-600">Tasks Today</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">96%</div>
                <div className="text-sm text-gray-600">Success Rate</div>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Agents Grid */}
        <div>
          <h2 className="text-2xl font-semibold mb-6">AI Agents</h2>
          {agentsLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="glass rounded-2xl p-6 animate-pulse">
                  <div className="h-4 bg-gray-200 rounded mb-2"></div>
                  <div className="h-3 bg-gray-200 rounded mb-4"></div>
                  <div className="space-y-2">
                    <div className="h-2 bg-gray-200 rounded"></div>
                    <div className="h-2 bg-gray-200 rounded"></div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {agents.map((agent) => (
                <AgentCard key={agent.id} agent={agent} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

## 🔧 Error Handling

### Global Error Boundary

```typescript
// components/ErrorBoundary.tsx
'use client';

import { Component, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-mesh">
          <div className="glass rounded-2xl p-8 max-w-md text-center">
            <h2 className="text-xl font-semibold text-red-600 mb-4">
              Something went wrong
            </h2>
            <p className="text-gray-600 mb-4">
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            <button
              onClick={() => this.setState({ hasError: false })}
              className="btn-gradient px-6 py-2 rounded-lg"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
```

## 🚀 Deployment Configuration

### Environment Variables

```bash
# Production .env.local
NEXT_PUBLIC_API_URL=https://your-api-domain.com
NEXT_PUBLIC_WS_URL=wss://your-api-domain.com/ws
```

### Next.js Configuration

```typescript
// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    domains: ['localhost', 'your-api-domain.com'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
```

## 📊 Performance Optimization

### Code Splitting

```typescript
// Dynamic imports for heavy components
import dynamic from 'next/dynamic';

const ChatInterface = dynamic(() => import('@/components/chat/ChatInterface'), {
  loading: () => <div className="glass rounded-2xl p-6 animate-pulse">Loading chat...</div>,
});

const AgentGrid = dynamic(() => import('@/components/agents/AgentGrid'), {
  loading: () => <div className="grid grid-cols-3 gap-6">Loading agents...</div>,
});
```

### Image Optimization

```typescript
import Image from 'next/image';

// Use Next.js Image component for optimized loading
<Image
  src="/agent-avatar.png"
  alt="Agent Avatar"
  width={48}
  height={48}
  className="rounded-full"
/>
```

## 🧪 Testing Integration

### API Testing

```typescript
// __tests__/api.test.ts
import { apiClient } from '@/utils/api';

describe('API Client', () => {
  test('should send chat message', async () => {
    const response = await apiClient.sendMessage('Hello');
    expect(response).toHaveProperty('response');
  });

  test('should search documents', async () => {
    const results = await apiClient.searchDocuments('FastAPI');
    expect(results).toHaveProperty('results');
  });
});
```

### Component Testing

```typescript
// __tests__/ChatInterface.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatInterface } from '@/components/chat/ChatInterface';

test('should send message on form submit', () => {
  render(<ChatInterface />);
  
  const input = screen.getByPlaceholderText('Ask me anything...');
  const button = screen.getByText('Send');
  
  fireEvent.change(input, { target: { value: 'Test message' } });
  fireEvent.click(button);
  
  expect(input).toHaveValue('');
});
```

---

This integration guide provides a complete foundation for connecting the Code Hero frontend with the FastAPI backend. The modular architecture ensures scalability and maintainability while the glassmorphism design provides a modern, engaging user experience. 