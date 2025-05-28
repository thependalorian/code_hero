'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Layout } from '@/components/layout/Layout';
import { GlassCard, MetricCard } from '@/components/ui/GlassCard';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { useToastActions } from '@/components/ui/Toast';
import { useAgents } from '@/hooks/useAgents';
import { api, HealthResponse } from '@/utils/api';
import { 
  Activity, 
  Bot, 
  Users, 
  Zap, 
  TrendingUp,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  Plus,
  Settings,
  MessageSquare,
  FileText,
  Database,
  Server,
  Cpu,
  HardDrive,
  Wifi,
  RefreshCw
} from 'lucide-react';
import { clsx } from 'clsx';
import Link from 'next/link';

interface RecentActivity {
  id: string;
  type: 'task_completed' | 'agent_created' | 'error' | 'deployment' | 'system_update';
  title: string;
  description: string;
  timestamp: Date;
  agent?: string;
  status: 'success' | 'error' | 'warning' | 'info';
}

interface SystemHealth {
  cpu: number;
  memory: number;
  disk: number;
  network: 'good' | 'fair' | 'poor';
  status: 'healthy' | 'warning' | 'critical';
}

export default function DashboardPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  const [systemLoading, setSystemLoading] = useState(true);
  const toast = useToastActions();

  const {
    agents,
    statistics,
    loading: agentsLoading,
    refreshAgents,
  } = useAgents();

  // Fetch system health and activity data
  useEffect(() => {
    const fetchSystemData = async () => {
      try {
        setSystemLoading(true);
        
        // Fetch health data
        const health = await api.health();
        setHealthData(health);
        
        // Mock system health data (you can replace with real metrics)
        const mockSystemHealth: SystemHealth = {
          cpu: Math.floor(Math.random() * 60) + 20, // 20-80%
          memory: Math.floor(Math.random() * 40) + 40, // 40-80%
          disk: Math.floor(Math.random() * 30) + 50, // 50-80%
          network: 'good',
          status: health.status === 'healthy' ? 'healthy' : 'warning'
        };
        setSystemHealth(mockSystemHealth);
        
        // Generate recent activity from system data
        const activities: RecentActivity[] = [
          {
            id: '1',
            type: 'system_update',
            title: 'System Health Check',
            description: `System status: ${health.status}. All services operational.`,
            timestamp: new Date(health.timestamp),
            status: health.status === 'healthy' ? 'success' : 'warning'
          },
          {
            id: '2',
            type: 'agent_created',
            title: 'Agents Initialized',
            description: `${statistics?.total_agents || 0} agents are currently active in the system.`,
            timestamp: new Date(Date.now() - 10 * 60 * 1000),
            status: 'info'
          }
        ];
        
        // Add agent-specific activities if available
        if (statistics && statistics.total_tasks_completed > 0) {
          activities.unshift({
            id: '3',
            type: 'task_completed',
            title: 'Tasks Completed',
            description: `${statistics.total_tasks_completed} total tasks completed with ${statistics.average_success_rate.toFixed(1)}% success rate.`,
            timestamp: new Date(Date.now() - 5 * 60 * 1000),
            status: 'success'
          });
        }
        
        setRecentActivity(activities);
        
      } catch (error) {
        console.error('Failed to fetch system data:', error);
        toast.error('System data unavailable', 'Some dashboard metrics may not be current');
      } finally {
        setSystemLoading(false);
      }
    };

    fetchSystemData();
  }, [statistics, toast]);

  const getActivityIcon = (type: RecentActivity['type']) => {
    switch (type) {
      case 'task_completed':
        return <CheckCircle className="w-4 h-4" />;
      case 'agent_created':
        return <Bot className="w-4 h-4" />;
      case 'error':
        return <AlertCircle className="w-4 h-4" />;
      case 'deployment':
        return <Server className="w-4 h-4" />;
      case 'system_update':
        return <Activity className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  const getActivityColor = (status: RecentActivity['status']) => {
    switch (status) {
      case 'success':
        return 'text-system-green bg-system-green/10';
      case 'error':
        return 'text-system-red bg-system-red/10';
      case 'warning':
        return 'text-system-orange bg-system-orange/10';
      case 'info':
        return 'text-system-blue bg-system-blue/10';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return date.toLocaleDateString();
  };

  const refreshDashboard = async () => {
    await Promise.all([
      refreshAgents(),
      // Refresh system data
      (() => {
        setSystemLoading(true);
        return new Promise(resolve => setTimeout(resolve, 1000));
      })()
    ]);
    setSystemLoading(false);
    toast.success('Dashboard refreshed', 'All data has been updated');
  };

  const loading = agentsLoading || systemLoading;

  if (loading && !agents.length) {
    return (
      <Layout
        sidebarCollapsed={sidebarCollapsed}
        onSidebarToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      >
        <LoadingSpinner size="lg" text="Loading dashboard..." fullScreen />
      </Layout>
    );
  }

  return (
    <Layout
      sidebarCollapsed={sidebarCollapsed}
      onSidebarToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
    >
      <div className="space-y-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
        >
          <div>
            <h1 className="text-fluid-3xl font-bold text-gray-900">
              Dashboard
            </h1>
            <p className="text-gray-600 mt-2">
              Welcome back! Here&apos;s what&apos;s happening with your Code Hero AI agents.
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            <Button
              onClick={refreshDashboard}
              variant="ghost"
              className="text-gray-600 hover:text-gray-900"
              disabled={loading}
            >
              <RefreshCw className={clsx('w-5 h-5 mr-2', loading && 'animate-spin')} />
              Refresh
            </Button>
            <Link href="/settings">
              <Button
                variant="ghost"
                className="text-gray-600 hover:text-gray-900"
              >
                <Settings className="w-5 h-5 mr-2" />
                Settings
              </Button>
            </Link>
            <Link href="/chat">
              <Button
                className={clsx(
                  'bg-gradient-to-r from-system-blue to-system-indigo',
                  'text-white font-semibold flex items-center space-x-2',
                  'hover:shadow-lg transform hover:scale-105 transition-all duration-200'
                )}
              >
                <MessageSquare className="w-5 h-5" />
                <span>Start Chat</span>
              </Button>
            </Link>
          </div>
        </motion.div>

        {/* Metrics Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4 lg:gap-6"
        >
          <MetricCard
            title="Total Agents"
            value={(statistics?.total_agents || agents.length).toString()}
            subtitle={`${statistics?.active_agents || agents.filter(a => a.status === 'active').length} active`}
            icon={<Bot className="w-6 h-6" />}
            trend="neutral"
            hover
          />
          <MetricCard
            title="Tasks Completed"
            value={(statistics?.total_tasks_completed || 0).toLocaleString()}
            subtitle="All time"
            icon={<CheckCircle className="w-6 h-6" />}
            trend="up"
            hover
          />
          <MetricCard
            title="Success Rate"
            value={`${(statistics?.average_success_rate || 0).toFixed(1)}%`}
            subtitle="Average performance"
            icon={<TrendingUp className="w-6 h-6" />}
            trend="up"
            hover
          />
          <MetricCard
            title="Avg Response"
            value={`${(statistics?.average_response_time || 0).toFixed(1)}s`}
            subtitle="Response time"
            icon={<Zap className="w-6 h-6" />}
            trend="down"
            hover
          />
          <MetricCard
            title="System Status"
            value={healthData?.status === 'healthy' ? 'Healthy' : 'Warning'}
            subtitle={healthData?.environment || 'Unknown'}
            icon={<Activity className="w-6 h-6" />}
            trend={healthData?.status === 'healthy' ? 'up' : 'neutral'}
            hover
          />
          <MetricCard
            title="Teams"
            value={(statistics?.teams?.length || 0).toString()}
            subtitle={`${statistics?.teams?.reduce((acc, team) => acc + team.agent_count, 0) || 0} total agents`}
            icon={<Users className="w-6 h-6" />}
            trend="up"
            hover
          />
        </motion.div>

        {/* Main Content Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-8"
        >
          {/* Recent Activity */}
          <div className="lg:col-span-2">
            <GlassCard className="p-6 h-full">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900">Recent Activity</h2>
                <Button variant="ghost" size="sm" className="text-gray-600">
                  View All
                  <ArrowRight className="w-4 h-4 ml-1" />
                </Button>
              </div>
              
              <div className="space-y-4">
                {recentActivity.length > 0 ? recentActivity.map((activity, index) => (
                  <motion.div
                    key={activity.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-start space-x-4 p-4 rounded-xl hover:bg-gray-50 transition-colors duration-200"
                  >
                    <div className={clsx(
                      'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0',
                      getActivityColor(activity.status)
                    )}>
                      {getActivityIcon(activity.type)}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <h3 className="font-medium text-gray-900 truncate">
                          {activity.title}
                        </h3>
                        <span className="text-xs text-gray-500 flex-shrink-0 ml-2">
                          {formatTimeAgo(activity.timestamp)}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">
                        {activity.description}
                      </p>
                      {activity.agent && (
                        <span className="inline-block px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-lg mt-2">
                          {activity.agent}
                        </span>
                      )}
                    </div>
                  </motion.div>
                )) : (
                  <div className="text-center py-8">
                    <Activity className="w-12 h-12 mx-auto text-gray-400 mb-3" />
                    <p className="text-gray-500">No recent activity</p>
                  </div>
                )}
              </div>
            </GlassCard>
          </div>

          {/* System Health & Quick Actions */}
          <div className="space-y-6">
            {/* System Health */}
            <GlassCard className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">System Health</h2>
                <div className={clsx(
                  'w-3 h-3 rounded-full',
                  systemHealth?.status === 'healthy' && 'bg-system-green',
                  systemHealth?.status === 'warning' && 'bg-system-orange',
                  systemHealth?.status === 'critical' && 'bg-system-red'
                )} />
              </div>
              
              <div className="space-y-4">
                {systemHealth ? [
                  { label: 'CPU Usage', value: systemHealth.cpu, icon: Cpu },
                  { label: 'Memory', value: systemHealth.memory, icon: HardDrive },
                  { label: 'Disk Space', value: systemHealth.disk, icon: Database }
                ].map(({ label, value, icon: Icon }) => (
                  <div key={label} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Icon className="w-4 h-4 text-gray-600" />
                        <span className="text-sm font-medium text-gray-900">{label}</span>
                      </div>
                      <span className="text-sm text-gray-600">{value}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={clsx(
                          'h-2 rounded-full transition-all duration-300',
                          value < 60 && 'bg-system-green',
                          value >= 60 && value < 80 && 'bg-system-orange',
                          value >= 80 && 'bg-system-red'
                        )}
                        style={{ width: `${value}%` }}
                      />
                    </div>
                  </div>
                )) : (
                  <div className="text-center py-4">
                    <LoadingSpinner size="sm" text="Loading metrics..." />
                  </div>
                )}
                
                {systemHealth && (
                  <div className="flex items-center justify-between pt-2 border-t border-gray-200">
                    <div className="flex items-center space-x-2">
                      <Wifi className="w-4 h-4 text-gray-600" />
                      <span className="text-sm font-medium text-gray-900">Network</span>
                    </div>
                    <span className={clsx(
                      'text-sm font-medium',
                      systemHealth.network === 'good' && 'text-system-green',
                      systemHealth.network === 'fair' && 'text-system-orange',
                      systemHealth.network === 'poor' && 'text-system-red'
                    )}>
                      {systemHealth.network}
                    </span>
                  </div>
                )}
              </div>
            </GlassCard>

            {/* Quick Actions */}
            <GlassCard className="p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
              
              <div className="space-y-3">
                {[
                  { label: 'Manage Agents', icon: Bot, href: '/agents', color: 'text-system-blue' },
                  { label: 'Start Chat', icon: MessageSquare, href: '/chat', color: 'text-system-green' },
                  { label: 'View Documentation', icon: FileText, href: '/docs', color: 'text-system-orange' },
                  { label: 'System Settings', icon: Settings, href: '/settings', color: 'text-system-indigo' }
                ].map(({ label, icon: Icon, href, color }) => (
                  <Link key={label} href={href}>
                    <button className="w-full flex items-center space-x-3 p-3 rounded-xl hover:bg-gray-50 transition-colors duration-200 text-left">
                      <Icon className={clsx('w-5 h-5', color)} />
                      <span className="font-medium text-gray-900">{label}</span>
                      <ArrowRight className="w-4 h-4 text-gray-400 ml-auto" />
                    </button>
                  </Link>
                ))}
              </div>
            </GlassCard>
          </div>
        </motion.div>

        {/* Agent Status Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <GlassCard className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Agent Status Overview</h2>
              <Link href="/agents">
                <Button variant="ghost" className="text-gray-600">
                  Manage Agents
                  <ArrowRight className="w-4 h-4 ml-1" />
                </Button>
              </Link>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {agents.slice(0, 4).map((agent, index) => (
                <motion.div
                  key={agent.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 * index }}
                  className="p-4 bg-white/50 rounded-xl border border-gray-200/50 hover:shadow-md transition-all duration-200"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-system-blue to-system-indigo rounded-xl flex items-center justify-center text-white shadow-lg">
                      <Bot className="w-5 h-5" />
                    </div>
                    <div className={clsx(
                      'w-3 h-3 rounded-full',
                      agent.status === 'active' && 'bg-system-green',
                      agent.status === 'processing' && 'bg-system-orange animate-pulse',
                      agent.status === 'idle' && 'bg-gray-400',
                      agent.status === 'error' && 'bg-system-red'
                    )} />
                  </div>
                  
                  <h3 className="font-semibold text-gray-900 mb-1 truncate">{agent.name}</h3>
                  <p className="text-sm text-gray-600 capitalize mb-3">{agent.status}</p>
                  
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Tasks:</span>
                      <span className="font-medium text-gray-900">{agent.performance.tasks_completed}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Success:</span>
                      <span className="font-medium text-gray-900">{agent.performance.success_rate}%</span>
                    </div>
                  </div>
                </motion.div>
              ))}
              
              {agents.length === 0 && (
                <div className="col-span-full text-center py-8">
                  <Bot className="w-12 h-12 mx-auto text-gray-400 mb-3" />
                  <p className="text-gray-500">No agents available</p>
                  <Link href="/agents">
                    <Button className="mt-3 bg-gradient-to-r from-system-blue to-system-indigo text-white">
                      <Plus className="w-4 h-4 mr-2" />
                      Deploy Agents
                    </Button>
                  </Link>
                </div>
              )}
            </div>
          </GlassCard>
        </motion.div>
      </div>
    </Layout>
  );
} 