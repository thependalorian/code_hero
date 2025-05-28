/**
 * HierarchicalAgentsPanel Component
 * 
 * Purpose: Displays and manages hierarchical agents system with real-time status,
 * agent interactions, and comprehensive monitoring capabilities.
 * 
 * Location: /components/agents/HierarchicalAgentsPanel.tsx
 * 
 * Features:
 * - Real-time agent status monitoring
 * - Interactive agent communication
 * - Hierarchical structure visualization
 * - Performance metrics display
 * - Error handling and logging
 */

'use client';

import { useState, useEffect, useRef } from 'react';
import { api, apiClient } from '@/utils/api';

// Types for agent system
interface Agent {
  id: string;
  name: string;
  role: string;
  status: 'active' | 'idle' | 'error' | 'offline';
  model: string;
  lastActivity: string;
  performance: {
    responseTime: number;
    successRate: number;
    totalRequests: number;
  };
}

interface AgentMessage {
  id: string;
  agentId: string;
  content: string;
  timestamp: string;
  type: 'request' | 'response' | 'error' | 'system';
}

interface HierarchicalSystem {
  supervisor: Agent;
  teams: {
    [teamName: string]: Agent[];
  };
  messages: AgentMessage[];
  systemStatus: 'healthy' | 'degraded' | 'error';
}

export default function HierarchicalAgentsPanel() {
  // State management
  const [system, setSystem] = useState<HierarchicalSystem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [userInput, setUserInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  
  // Refs for real-time updates
  const refreshInterval = useRef<NodeJS.Timeout | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch system status
  const fetchSystemStatus = async () => {
    try {
      const response = await api.agents.getAll();
      const hierarchicalStatus = await apiClient.getHierarchicalSystemStatus();
      
      // Transform the data to match our interface
      const transformedSystem: HierarchicalSystem = {
        supervisor: {
          id: 'supervisor',
          name: 'Supervisor Agent',
          role: 'Supervisor',
          status: 'active',
          model: 'gpt-4o',
          lastActivity: new Date().toISOString(),
          performance: {
            responseTime: 150,
            successRate: 98,
            totalRequests: 1250
          }
        },
        teams: {
          'Development': response.filter(agent => agent.team === 'Development').map(agent => ({
            id: agent.id,
            name: agent.name,
            role: agent.type,
            status: agent.status as 'active' | 'idle' | 'error' | 'offline',
            model: 'gpt-4o',
            lastActivity: agent.last_active,
            performance: {
              responseTime: 200,
              successRate: agent.performance.success_rate,
              totalRequests: agent.performance.tasks_completed
            }
          })),
          'Research': response.filter(agent => agent.team === 'Research').map(agent => ({
            id: agent.id,
            name: agent.name,
            role: agent.type,
            status: agent.status as 'active' | 'idle' | 'error' | 'offline',
            model: 'gpt-4-turbo-preview',
            lastActivity: agent.last_active,
            performance: {
              responseTime: 180,
              successRate: agent.performance.success_rate,
              totalRequests: agent.performance.tasks_completed
            }
          }))
        },
        messages: [],
        systemStatus: hierarchicalStatus.status === 'healthy' ? 'healthy' : 'degraded'
      };
      
      setSystem(transformedSystem);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch system status:', err);
      setError('Failed to connect to hierarchical agents system');
    } finally {
      setLoading(false);
    }
  };

  // Send message to hierarchical system
  const sendMessage = async () => {
    if (!userInput.trim() || isProcessing) return;

    setIsProcessing(true);
    try {
      const response = await apiClient.processWithHierarchicalAgents(
        userInput,
        undefined, // conversationId
        undefined, // projectId
        'medium', // taskPriority
        {
          selectedAgent: selectedAgent?.id,
          timestamp: new Date().toISOString()
        }
      );

      // Update system with new message
      if (system) {
        const newMessage: AgentMessage = {
          id: Date.now().toString(),
          agentId: 'user',
          content: userInput,
          timestamp: new Date().toISOString(),
          type: 'request'
        };

        const responseMessage: AgentMessage = {
          id: (Date.now() + 1).toString(),
          agentId: response.agents_used[0] || 'system',
          content: response.response,
          timestamp: new Date().toISOString(),
          type: 'response'
        };

        setSystem(prev => prev ? {
          ...prev,
          messages: [...prev.messages, newMessage, responseMessage]
        } : null);
      }

      setUserInput('');
    } catch (err) {
      console.error('Failed to send message:', err);
      setError('Failed to process message');
    } finally {
      setIsProcessing(false);
    }
  };

  // Auto-scroll to latest messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [system?.messages]);

  // Auto-refresh system status
  useEffect(() => {
    fetchSystemStatus();

    if (autoRefresh) {
      refreshInterval.current = setInterval(fetchSystemStatus, 5000);
    }

    return () => {
      if (refreshInterval.current) {
        clearInterval(refreshInterval.current);
      }
    };
  }, [autoRefresh]);

  // Get status color for agent
  const getStatusColor = (status: Agent['status']) => {
    switch (status) {
      case 'active': return 'badge-success';
      case 'idle': return 'badge-warning';
      case 'error': return 'badge-error';
      case 'offline': return 'badge-neutral';
      default: return 'badge-neutral';
    }
  };

  // Get system status color
  const getSystemStatusColor = (status: HierarchicalSystem['systemStatus']) => {
    switch (status) {
      case 'healthy': return 'text-success';
      case 'degraded': return 'text-warning';
      case 'error': return 'text-error';
      default: return 'text-neutral';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="loading loading-spinner loading-lg"></div>
        <span className="ml-4 text-lg">Loading Hierarchical Agents System...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alert alert-error">
        <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span>{error}</span>
        <div>
          <button className="btn btn-sm btn-outline" onClick={fetchSystemStatus}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!system) {
    return (
      <div className="alert alert-warning">
        <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16c-.77.833.192 2.5 1.732 2.5z" />
        </svg>
        <span>No hierarchical agents system found</span>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Hierarchical Agents System</h1>
          <p className="text-base-content/70 mt-2">
            Monitor and interact with the multi-agent hierarchical system
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          <div className={`text-lg font-semibold ${getSystemStatusColor(system.systemStatus)}`}>
            System: {system.systemStatus.toUpperCase()}
          </div>
          
          <div className="form-control">
            <label className="label cursor-pointer">
              <span className="label-text mr-2">Auto-refresh</span>
              <input 
                type="checkbox" 
                className="toggle toggle-primary" 
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            </label>
          </div>
          
          <button 
            className="btn btn-primary btn-sm"
            onClick={fetchSystemStatus}
            disabled={loading}
          >
            {loading ? <span className="loading loading-spinner loading-xs"></span> : 'Refresh'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Agents Panel */}
        <div className="lg:col-span-1 space-y-4">
          {/* Supervisor */}
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <h2 className="card-title text-primary">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Supervisor Agent
              </h2>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="font-medium">{system.supervisor.name}</span>
                  <div className={`badge ${getStatusColor(system.supervisor.status)}`}>
                    {system.supervisor.status}
                  </div>
                </div>
                
                <div className="text-sm space-y-1">
                  <div>Model: <span className="font-mono">{system.supervisor.model}</span></div>
                  <div>Role: <span className="text-primary">{system.supervisor.role}</span></div>
                  <div>Last Activity: {new Date(system.supervisor.lastActivity).toLocaleTimeString()}</div>
                </div>
                
                <div className="stats stats-vertical shadow">
                  <div className="stat">
                    <div className="stat-title">Response Time</div>
                    <div className="stat-value text-sm">{system.supervisor.performance.responseTime}ms</div>
                  </div>
                  <div className="stat">
                    <div className="stat-title">Success Rate</div>
                    <div className="stat-value text-sm">{system.supervisor.performance.successRate}%</div>
                  </div>
                  <div className="stat">
                    <div className="stat-title">Total Requests</div>
                    <div className="stat-value text-sm">{system.supervisor.performance.totalRequests}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Teams */}
          {Object.entries(system.teams).map(([teamName, agents]) => (
            <div key={teamName} className="card bg-base-100 shadow-xl">
              <div className="card-body">
                <h3 className="card-title text-secondary">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                  {teamName} Team ({agents.length})
                </h3>
                
                <div className="space-y-2">
                  {agents.map((agent) => (
                    <div 
                      key={agent.id}
                      className={`p-3 rounded-lg border cursor-pointer transition-all hover:shadow-md ${
                        selectedAgent?.id === agent.id ? 'border-primary bg-primary/10' : 'border-base-300'
                      }`}
                      onClick={() => setSelectedAgent(agent)}
                    >
                      <div className="flex justify-between items-center">
                        <span className="font-medium">{agent.name}</span>
                        <div className={`badge badge-sm ${getStatusColor(agent.status)}`}>
                          {agent.status}
                        </div>
                      </div>
                      <div className="text-xs text-base-content/70 mt-1">
                        {agent.role} • {agent.model}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Messages and Interaction Panel */}
        <div className="lg:col-span-2 space-y-4">
          {/* Messages */}
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <h2 className="card-title">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-3.582 8-8 8a8.959 8.959 0 01-4.906-1.436L3 21l2.436-5.094A8.959 8.959 0 013 12c0-4.418 3.582-8 8-8s8 3.582 8 8z" />
                </svg>
                Agent Communications
                <div className="badge badge-primary">{system.messages.length}</div>
              </h2>
              
              <div className="h-96 overflow-y-auto space-y-3 p-4 bg-base-200 rounded-lg">
                {system.messages.length === 0 ? (
                  <div className="text-center text-base-content/50 py-8">
                    No messages yet. Start a conversation with the agents!
                  </div>
                ) : (
                  system.messages.map((message) => (
                    <div 
                      key={message.id}
                      className={`chat ${message.agentId === 'user' ? 'chat-end' : 'chat-start'}`}
                    >
                      <div className="chat-image avatar">
                        <div className="w-10 rounded-full bg-primary text-primary-content flex items-center justify-center">
                          {message.agentId === 'user' ? 'U' : 'A'}
                        </div>
                      </div>
                      <div className="chat-header">
                        {message.agentId === 'user' ? 'You' : message.agentId}
                        <time className="text-xs opacity-50 ml-2">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </time>
                      </div>
                      <div className={`chat-bubble ${
                        message.type === 'error' ? 'chat-bubble-error' :
                        message.type === 'system' ? 'chat-bubble-info' :
                        message.agentId === 'user' ? 'chat-bubble-primary' : 'chat-bubble-secondary'
                      }`}>
                        {message.content}
                      </div>
                    </div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>
          </div>

          {/* Input Panel */}
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <h3 className="card-title text-sm">Send Message to Hierarchical System</h3>
              
              {selectedAgent && (
                <div className="alert alert-info">
                  <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Targeting: {selectedAgent.name} ({selectedAgent.role})</span>
                  <button 
                    className="btn btn-sm btn-ghost"
                    onClick={() => setSelectedAgent(null)}
                  >
                    Clear
                  </button>
                </div>
              )}
              
              <div className="form-control">
                <div className="input-group">
                  <textarea
                    className="textarea textarea-bordered flex-1 min-h-[100px]"
                    placeholder="Enter your message for the hierarchical agents system..."
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    disabled={isProcessing}
                  />
                  <button 
                    className="btn btn-primary"
                    onClick={sendMessage}
                    disabled={!userInput.trim() || isProcessing}
                  >
                    {isProcessing ? (
                      <span className="loading loading-spinner loading-sm"></span>
                    ) : (
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                      </svg>
                    )}
                  </button>
                </div>
                <div className="label">
                  <span className="label-text-alt">Press Enter to send, Shift+Enter for new line</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 