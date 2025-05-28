'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  Bot, 
  Zap, 
  Clock, 
  CheckCircle, 
  XCircle,
  Activity,
  Brain,
  Code,
  Search,
  FileText
} from 'lucide-react';
import { StatusCard } from '../ui/GlassCard';
import { Button } from '../ui/Button';
import { clsx } from 'clsx';

interface Agent {
  id: string;
  name: string;
  type: 'research' | 'coding' | 'strategic' | 'langchain' | 'langgraph' | 'llamaindex' | 'fastapi' | 'nextjs' | 'pydantic' | 'agno' | 'crewai' | 'supervisor' | 'prompt_engineer' | 'implementation' | 'documentation' | 'trd_converter' | 'code_generator' | 'code_reviewer' | 'standards_enforcer' | 'document_analyzer';
  status: 'active' | 'processing' | 'idle' | 'error';
  description: string;
  capabilities: string[];
  performance: {
    tasksCompleted: number;
    successRate: number;
    avgResponseTime: string;
    uptime: string;
  };
  currentTask?: string;
  lastActive: Date;
}

interface AgentCardProps {
  agent: Agent;
  onInteract?: (agentId: string) => void;
  onViewDetails?: (agentId: string) => void;
  className?: string;
}

const agentIcons = {
  research: Search,
  coding: Code,
  strategic: Brain,
  langchain: Bot,
  langgraph: Bot,
  llamaindex: Search,
  fastapi: Zap,
  nextjs: FileText,
  pydantic: Code,
  agno: Bot,
  crewai: Bot,
  supervisor: Brain,
  prompt_engineer: FileText,
  implementation: Code,
  documentation: FileText,
  trd_converter: FileText,
  code_generator: Code,
  code_reviewer: CheckCircle,
  standards_enforcer: CheckCircle,
  document_analyzer: Search
};

const agentGradients = {
  research: 'from-blue-500 to-cyan-500',
  coding: 'from-green-500 to-emerald-500',
  strategic: 'from-purple-500 to-pink-500',
  langchain: 'from-orange-500 to-red-500',
  langgraph: 'from-orange-600 to-red-600',
  llamaindex: 'from-blue-600 to-cyan-600',
  fastapi: 'from-indigo-500 to-blue-500',
  nextjs: 'from-gray-700 to-gray-900',
  pydantic: 'from-green-600 to-emerald-600',
  agno: 'from-purple-600 to-pink-600',
  crewai: 'from-yellow-500 to-orange-500',
  supervisor: 'from-red-500 to-pink-500',
  prompt_engineer: 'from-teal-500 to-cyan-500',
  implementation: 'from-green-700 to-emerald-700',
  documentation: 'from-blue-700 to-indigo-700',
  trd_converter: 'from-purple-700 to-pink-700',
  code_generator: 'from-green-800 to-emerald-800',
  code_reviewer: 'from-yellow-600 to-orange-600',
  standards_enforcer: 'from-red-600 to-pink-600',
  document_analyzer: 'from-blue-800 to-cyan-800'
};

export const AgentCard: React.FC<AgentCardProps> = ({
  agent,
  onInteract,
  onViewDetails,
  className = ''
}) => {
  const IconComponent = agentIcons[agent.type];
  const gradient = agentGradients[agent.type];

  const getStatusIcon = () => {
    switch (agent.status) {
      case 'active':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'processing':
        return <Activity className="w-4 h-4 text-yellow-500 animate-pulse" />;
      case 'error':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusText = () => {
    switch (agent.status) {
      case 'active':
        return 'Active';
      case 'processing':
        return 'Processing';
      case 'error':
        return 'Error';
      default:
        return 'Idle';
    }
  };

  return (
    <StatusCard
      status={agent.status}
      hover={true}
      className={clsx('h-full', className)}
    >
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className={`w-12 h-12 bg-gradient-to-r ${gradient} rounded-xl flex items-center justify-center shadow-lg`}>
              <IconComponent className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">{agent.name}</h3>
              <p className="text-sm text-gray-500 capitalize">{agent.type} Agent</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {getStatusIcon()}
            <span className={clsx(
              'text-xs font-medium px-2 py-1 rounded-full',
              agent.status === 'active' && 'bg-green-100 text-green-700',
              agent.status === 'processing' && 'bg-yellow-100 text-yellow-700',
              agent.status === 'error' && 'bg-red-100 text-red-700',
              agent.status === 'idle' && 'bg-gray-100 text-gray-700'
            )}>
              {getStatusText()}
            </span>
          </div>
        </div>

        {/* Description */}
        <p className="text-gray-600 text-sm leading-relaxed">
          {agent.description}
        </p>

        {/* Current Task */}
        {agent.currentTask && (
          <div className="glass rounded-lg p-3">
            <p className="text-xs font-medium text-gray-500 mb-1">Current Task</p>
            <p className="text-sm text-gray-900">{agent.currentTask}</p>
          </div>
        )}

        {/* Capabilities */}
        <div>
          <p className="text-xs font-medium text-gray-500 mb-2">Capabilities</p>
          <div className="flex flex-wrap gap-1">
            {agent.capabilities.slice(0, 3).map((capability, index) => (
              <span
                key={index}
                className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full"
              >
                {capability}
              </span>
            ))}
            {agent.capabilities.length > 3 && (
              <span className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded-full">
                +{agent.capabilities.length - 3} more
              </span>
            )}
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-2 gap-3">
          <div className="text-center">
            <p className="text-lg font-bold text-gray-900">{agent.performance.tasksCompleted}</p>
            <p className="text-xs text-gray-500">Tasks</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold text-gray-900">{agent.performance.successRate}%</p>
            <p className="text-xs text-gray-500">Success</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold text-gray-900">{agent.performance.avgResponseTime}</p>
            <p className="text-xs text-gray-500">Avg Time</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold text-gray-900">{agent.performance.uptime}</p>
            <p className="text-xs text-gray-500">Uptime</p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex space-x-2 pt-2">
          <Button
            variant="primary"
            size="sm"
            className="flex-1"
            onClick={() => onInteract?.(agent.id)}
            disabled={agent.status === 'error'}
          >
            Interact
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onViewDetails?.(agent.id)}
          >
            Details
          </Button>
        </div>

        {/* Last Active */}
        <p className="text-xs text-gray-400 text-center">
          Last active: {agent.lastActive.toLocaleString()}
        </p>
      </div>
    </StatusCard>
  );
};

export const AgentGrid: React.FC<{
  agents: Agent[];
  onInteract?: (agentId: string) => void;
  onViewDetails?: (agentId: string) => void;
  className?: string;
}> = ({ agents, onInteract, onViewDetails, className = '' }) => {
  return (
    <div className={clsx('grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6', className)}>
      {agents.map((agent, index) => (
        <motion.div
          key={agent.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <AgentCard
            agent={agent}
            onInteract={onInteract}
            onViewDetails={onViewDetails}
          />
        </motion.div>
      ))}
    </div>
  );
}; 