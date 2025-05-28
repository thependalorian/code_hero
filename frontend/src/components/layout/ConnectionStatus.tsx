/**
 * Connection Status Component
 * Shows real-time backend connection status
 */

'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { WifiOff, AlertCircle, CheckCircle } from 'lucide-react';
import { useHealth } from '@/hooks/useHealth';

interface ConnectionStatusProps {
  className?: string;
  showDetails?: boolean;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  className = '',
  showDetails = false
}) => {
  const { isConnected, health, error, lastChecked } = useHealth({
    pollInterval: 15000, // Check every 15 seconds
    autoStart: true
  });

  const getStatusColor = () => {
    if (isConnected) return 'text-green-600 bg-green-50 border-green-200';
    if (error) return 'text-red-600 bg-red-50 border-red-200';
    return 'text-yellow-600 bg-yellow-50 border-yellow-200';
  };

  const getStatusIcon = () => {
    if (isConnected) return <CheckCircle className="w-4 h-4" />;
    if (error) return <AlertCircle className="w-4 h-4" />;
    return <WifiOff className="w-4 h-4" />;
  };

  const getStatusText = () => {
    if (isConnected) return 'Connected';
    if (error) return 'Error';
    return 'Connecting...';
  };

  return (
    <div className={`${className}`}>
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm border ${getStatusColor()}`}
      >
        {getStatusIcon()}
        <span className="font-medium">{getStatusText()}</span>
        
        {lastChecked && (
          <span className="text-xs opacity-70">
            {lastChecked.toLocaleTimeString([], { 
              hour: '2-digit', 
              minute: '2-digit' 
            })}
          </span>
        )}
      </motion.div>

      {/* Detailed Status */}
      <AnimatePresence>
        {showDetails && health && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full left-0 mt-2 p-3 bg-white rounded-lg shadow-lg border border-gray-200 z-50 min-w-64"
          >
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Backend Status</span>
                <span className={`text-xs px-2 py-1 rounded ${
                  health.status === 'healthy' 
                    ? 'bg-green-100 text-green-700' 
                    : 'bg-red-100 text-red-700'
                }`}>
                  {health.status}
                </span>
              </div>
              
              <div className="text-xs text-gray-600 space-y-1">
                <div>Environment: {health.environment}</div>
                <div>Services: {Object.keys(health.services).length} active</div>
                <div>
                  State Manager: {health.services.state_manager.project_count} projects, {' '}
                  {health.services.state_manager.chat_count} chats
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}; 