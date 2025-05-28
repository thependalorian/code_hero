/**
 * React hook for monitoring backend health
 * Provides real-time status of backend services
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { apiClient, type HealthResponse } from '@/utils/api';

interface UseHealthOptions {
  pollInterval?: number; // in milliseconds
  autoStart?: boolean;
}

interface UseHealthReturn {
  health: HealthResponse | null;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  lastChecked: Date | null;
  checkHealth: () => Promise<void>;
  startPolling: () => void;
  stopPolling: () => void;
}

export function useHealth(options: UseHealthOptions = {}): UseHealthReturn {
  const { pollInterval = 30000, autoStart = true } = options; // Default: check every 30 seconds
  
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const checkHealth = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const healthData = await apiClient.healthCheck();
      setHealth(healthData);
      setIsConnected(healthData.status === 'healthy');
      setLastChecked(new Date());
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Health check failed';
      setError(errorMessage);
      setIsConnected(false);
      setHealth(null);
      console.error('Health check failed:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const startPolling = useCallback(() => {
    // Clear existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    // Initial check
    checkHealth();

    // Set up polling
    intervalRef.current = setInterval(checkHealth, pollInterval);
  }, [checkHealth, pollInterval]);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // Auto-start polling on mount
  useEffect(() => {
    if (autoStart) {
      startPolling();
    }

    // Cleanup on unmount
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [autoStart, startPolling]);

  return {
    health,
    isConnected,
    isLoading,
    error,
    lastChecked,
    checkHealth,
    startPolling,
    stopPolling,
  };
} 