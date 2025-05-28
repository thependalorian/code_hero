'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, AlertCircle } from 'lucide-react';
import { AgentGrid } from './AgentCard';
import { Button } from '../ui/Button';

interface BackendAgent {
  id: string;
  name: string;
  type: string;
  status: string;
  description: string;
  capabilities: string[];
  performance: {
    tasks_completed: number;
    success_rate: number;
    avg_response_time: string;
    uptime: string;
  };
  current_task?: string;
  last_active: string;
}

interface Agent {
  id: string;
  name: string;
  type: 'research' | 'coding' | 'strategic' | 'langchain' | 'langgraph' | 'llamaindex' | 'fastapi' | 'nextjs' | 'pydantic' | 'agno' | 'crewai' | 'supervisor' | 'prompt_engineer' | 'implementation' | 'documentation' | 'trd_converter' | 'code_generator' | 'code_reviewer' | 'standards_enforcer' | 'document_analyzer';
  status: 'active' | 'processing' | 'idle' | 'error';
  description: string;
  capabilities: string[];
  performance: {
    tasks_completed: number;
    success_rate: number;
    avg_response_time: string;
    uptime: string;
  };
  current_task?: string;
  last_active: string;
}

interface AgentStatistics {
  total_agents: number;
  active_agents: number;
  processing_agents: number;
  idle_agents: number;
  error_agents: number;
  total_tasks_completed: number;
  overall_success_rate: number;
}

interface TaskHistory {
  task: string;
  success: boolean;
  duration: number;
}

interface AgentHistoryResponse {
  total_tasks: number;
  recent_tasks: TaskHistory[];
}

export const AgentDashboard: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [statistics, setStatistics] = useState<AgentStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchAgents = async () => {
    try {
      const response = await fetch('/api/agents/');
      if (!response.ok) {
        throw new Error(`Failed to fetch agents: ${response.statusText}`);
      }
      const data = await response.json();
      
      // Transform backend data to frontend format
      const transformedAgents: Agent[] = data.map((agent: BackendAgent) => ({
        id: agent.id,
        name: agent.name,
        type: agent.type as Agent['type'],
        status: agent.status as Agent['status'],
        description: agent.description,
        capabilities: agent.capabilities,
        performance: {
          tasks_completed: agent.performance.tasks_completed,
          success_rate: agent.performance.success_rate,
          avg_response_time: agent.performance.avg_response_time,
          uptime: agent.performance.uptime
        },
        current_task: agent.current_task,
        last_active: agent.last_active
      }));
      
      setAgents(transformedAgents);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch agents');
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await fetch('/api/agents/statistics/overview');
      if (!response.ok) {
        throw new Error(`Failed to fetch statistics: ${response.statusText}`);
      }
      const data = await response.json();
      setStatistics(data);
    } catch (err) {
      console.error('Failed to fetch statistics:', err);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([fetchAgents(), fetchStatistics()]);
    setRefreshing(false);
  };

  const handleAgentInteract = async (agentId: string) => {
    try {
      const message = prompt('Enter a message for the agent:');
      if (!message) return;

      const response = await fetch(`/api/agents/${agentId}/interact?message=${encodeURIComponent(message)}`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`Failed to interact with agent: ${response.statusText}`);
      }

      const result = await response.json();
      alert(`Agent Response:\n\n${result.response}`);
      
      // Refresh agents to see updated status
      await handleRefresh();
    } catch (err) {
      alert(`Error: ${err instanceof Error ? err.message : 'Failed to interact with agent'}`);
    }
  };

  const handleViewDetails = async (agentId: string) => {
    try {
      const response = await fetch(`/api/agents/${agentId}/history?limit=5`);
      if (!response.ok) {
        throw new Error(`Failed to fetch agent history: ${response.statusText}`);
      }

      const data: AgentHistoryResponse = await response.json();
      const historyText = data.recent_tasks.length > 0 
        ? data.recent_tasks.map((task: TaskHistory, index: number) => 
            `${index + 1}. ${task.task} (${task.success ? 'Success' : 'Failed'}) - ${task.duration}s`
          ).join('\n')
        : 'No recent tasks';

      alert(`Agent History (${data.total_tasks} total tasks):\n\n${historyText}`);
    } catch (err) {
      alert(`Error: ${err instanceof Error ? err.message : 'Failed to fetch agent details'}`);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchAgents(), fetchStatistics()]);
      setLoading(false);
    };

    loadData();

    // Set up auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchAgents();
      fetchStatistics();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-500" />
          <p className="text-gray-600">Loading agents...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <AlertCircle className="w-8 h-8 mx-auto mb-4 text-red-500" />
          <p className="text-red-600 mb-4">{error}</p>
          <Button onClick={handleRefresh} variant="primary">
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Agent Dashboard</h1>
          <p className="text-gray-600 mt-2">Monitor and interact with your AI agents</p>
        </div>
        <Button
          onClick={handleRefresh}
          variant="ghost"
          className="flex items-center space-x-2"
          disabled={refreshing}
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </Button>
      </div>

      {/* Statistics */}
      {statistics && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4 mb-8"
        >
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <p className="text-2xl font-bold text-gray-900">{statistics.total_agents}</p>
            <p className="text-sm text-gray-500">Total Agents</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <p className="text-2xl font-bold text-green-600">{statistics.active_agents}</p>
            <p className="text-sm text-gray-500">Active</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <p className="text-2xl font-bold text-yellow-600">{statistics.processing_agents}</p>
            <p className="text-sm text-gray-500">Processing</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <p className="text-2xl font-bold text-gray-600">{statistics.idle_agents}</p>
            <p className="text-sm text-gray-500">Idle</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <p className="text-2xl font-bold text-red-600">{statistics.error_agents}</p>
            <p className="text-sm text-gray-500">Error</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <p className="text-2xl font-bold text-blue-600">{statistics.total_tasks_completed}</p>
            <p className="text-sm text-gray-500">Tasks</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <p className="text-2xl font-bold text-purple-600">{statistics.overall_success_rate.toFixed(1)}%</p>
            <p className="text-sm text-gray-500">Success Rate</p>
          </div>
        </motion.div>
      )}

      {/* Agents Grid */}
      <AgentGrid
        agents={agents.map(agent => ({
          id: agent.id,
          name: agent.name,
          type: agent.type,
          status: agent.status,
          description: agent.description,
          capabilities: agent.capabilities,
          performance: {
            tasksCompleted: agent.performance.tasks_completed,
            successRate: agent.performance.success_rate,
            avgResponseTime: agent.performance.avg_response_time,
            uptime: agent.performance.uptime
          },
          currentTask: agent.current_task,
          lastActive: new Date(agent.last_active)
        }))}
        onInteract={handleAgentInteract}
        onViewDetails={handleViewDetails}
      />
    </div>
  );
}; 