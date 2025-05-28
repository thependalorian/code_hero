'use client';

import { useState, useEffect, useCallback } from 'react';
import { api, Agent, AgentStatistics, AgentHistoryResponse } from '@/utils/api';
import { useToastActions } from '@/components/ui/Toast';

interface UseAgentsReturn {
  agents: Agent[];
  statistics: AgentStatistics | null;
  loading: boolean;
  error: string | null;
  refreshAgents: () => Promise<void>;
  interactWithAgent: (agentId: string, message: string) => Promise<string>;
  updateAgentStatus: (agentId: string, status: Agent['status'], currentTask?: string) => Promise<void>;
  getAgentHistory: (agentId: string, limit?: number) => Promise<AgentHistoryResponse>;
}

export const useAgents = (): UseAgentsReturn => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [statistics, setStatistics] = useState<AgentStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const toast = useToastActions();

  const fetchAgents = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [agentsData, statsData] = await Promise.all([
        api.agents.getAll(),
        api.agents.getStatistics()
      ]);
      
      setAgents(agentsData);
      setStatistics(statsData);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch agents';
      setError(errorMessage);
      console.error('Failed to fetch agents:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshAgents = useCallback(async () => {
    await fetchAgents();
  }, [fetchAgents]);

  const interactWithAgent = useCallback(async (agentId: string, message: string): Promise<string> => {
    try {
      const response = await api.agents.interact(agentId, message);
      
      if (response.success) {
        toast.success('Agent interaction successful', `Response received in ${response.duration}`);
        // Refresh agents to update status
        await refreshAgents();
        return response.response;
      } else {
        throw new Error('Agent interaction failed');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to interact with agent';
      toast.error('Agent interaction failed', errorMessage);
      throw err;
    }
  }, [toast, refreshAgents]);

  const updateAgentStatus = useCallback(async (
    agentId: string, 
    status: Agent['status'], 
    currentTask?: string
  ) => {
    try {
      await api.agents.updateStatus(agentId, status, currentTask);
      toast.success('Agent status updated', `Agent ${agentId} is now ${status}`);
      // Refresh agents to reflect the change
      await refreshAgents();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update agent status';
      toast.error('Status update failed', errorMessage);
      throw err;
    }
  }, [toast, refreshAgents]);

  const getAgentHistory = useCallback(async (agentId: string, limit: number = 10) => {
    try {
      return await api.agents.getHistory(agentId, limit);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get agent history';
      toast.error('Failed to load history', errorMessage);
      throw err;
    }
  }, [toast]);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  return {
    agents,
    statistics,
    loading,
    error,
    refreshAgents,
    interactWithAgent,
    updateAgentStatus,
    getAgentHistory,
  };
}; 