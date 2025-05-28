'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  FolderOpen, 
  Calendar, 
  GitBranch, 
  Clock, 
  CheckCircle,
  AlertTriangle,
  Play,
  Pause,
  MoreHorizontal
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import { Button } from '../ui/Button';
import { clsx } from 'clsx';
import Image from 'next/image';

interface Project {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'paused' | 'completed' | 'planning';
  progress: number;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  dueDate: Date;
  createdAt: Date;
  team: {
    id: string;
    name: string;
    avatar?: string;
    role: string;
  }[];
  agents: string[];
  technologies: string[];
  repository?: string;
  lastActivity: Date;
}

interface ProjectCardProps {
  project: Project;
  onOpen?: (projectId: string) => void;
  onEdit?: (projectId: string) => void;
  onDelete?: (projectId: string) => void;
  className?: string;
}

const priorityColors = {
  low: 'bg-gray-100 text-gray-700',
  medium: 'bg-blue-100 text-blue-700',
  high: 'bg-orange-100 text-orange-700',
  urgent: 'bg-red-100 text-red-700'
};

const statusColors = {
  active: 'bg-green-100 text-green-700',
  paused: 'bg-yellow-100 text-yellow-700',
  completed: 'bg-blue-100 text-blue-700',
  planning: 'bg-purple-100 text-purple-700'
};

const statusIcons = {
  active: Play,
  paused: Pause,
  completed: CheckCircle,
  planning: Clock
};

export const ProjectCard: React.FC<ProjectCardProps> = ({
  project,
  onOpen,
  onEdit,
  onDelete: _onDelete, // eslint-disable-line @typescript-eslint/no-unused-vars
  className = ''
}) => {
  const StatusIcon = statusIcons[project.status];
  const daysUntilDue = Math.ceil((project.dueDate.getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24));
  const isOverdue = daysUntilDue < 0;
  const isDueSoon = daysUntilDue <= 3 && daysUntilDue >= 0;

  return (
    <GlassCard hover={true} className={clsx('h-full', className)}>
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-primary rounded-xl flex items-center justify-center shadow-lg">
              <FolderOpen className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-gray-900 line-clamp-1">{project.name}</h3>
              <div className="flex items-center space-x-2 mt-1">
                <StatusIcon className="w-3 h-3" />
                <span className={clsx(
                  'text-xs font-medium px-2 py-1 rounded-full',
                  statusColors[project.status]
                )}>
                  {project.status.charAt(0).toUpperCase() + project.status.slice(1)}
                </span>
                <span className={clsx(
                  'text-xs font-medium px-2 py-1 rounded-full',
                  priorityColors[project.priority]
                )}>
                  {project.priority.charAt(0).toUpperCase() + project.priority.slice(1)}
                </span>
              </div>
            </div>
          </div>
          
          <Button variant="ghost" size="sm">
            <MoreHorizontal className="w-4 h-4" />
          </Button>
        </div>

        {/* Description */}
        <p className="text-gray-600 text-sm leading-relaxed line-clamp-2">
          {project.description}
        </p>

        {/* Progress */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-gray-500">Progress</span>
            <span className="text-xs font-medium text-gray-900">{project.progress}%</span>
          </div>
          <div className="progress-bar">
            <motion.div
              className="progress-fill"
              initial={{ width: 0 }}
              animate={{ width: `${project.progress}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
            />
          </div>
        </div>

        {/* Due Date */}
        <div className="flex items-center space-x-2">
          <Calendar className="w-4 h-4 text-gray-400" />
          <span className={clsx(
            'text-sm',
            isOverdue && 'text-red-600 font-medium',
            isDueSoon && 'text-orange-600 font-medium',
            !isOverdue && !isDueSoon && 'text-gray-600'
          )}>
            Due {project.dueDate.toLocaleDateString()}
            {isOverdue && ` (${Math.abs(daysUntilDue)} days overdue)`}
            {isDueSoon && ` (${daysUntilDue} days left)`}
          </span>
          {(isOverdue || isDueSoon) && (
            <AlertTriangle className={clsx(
              'w-4 h-4',
              isOverdue ? 'text-red-500' : 'text-orange-500'
            )} />
          )}
        </div>

        {/* Team Members */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-gray-500">Team</span>
            <span className="text-xs text-gray-400">{project.team.length} members</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="flex -space-x-2">
              {project.team.slice(0, 4).map((member) => (
                <div
                  key={member.id}
                  className="w-8 h-8 bg-gradient-accent rounded-full flex items-center justify-center text-white text-xs font-medium border-2 border-white shadow-sm"
                  title={member.name}
                >
                  {member.avatar ? (
                    <Image src={member.avatar} alt={member.name} width={32} height={32} className="w-full h-full rounded-full object-cover" />
                  ) : (
                    member.name.charAt(0).toUpperCase()
                  )}
                </div>
              ))}
              {project.team.length > 4 && (
                <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center text-gray-600 text-xs font-medium border-2 border-white shadow-sm">
                  +{project.team.length - 4}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* AI Agents */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-gray-500">AI Agents</span>
            <span className="text-xs text-gray-400">{project.agents.length} active</span>
          </div>
          <div className="flex flex-wrap gap-1">
            {project.agents.slice(0, 3).map((agent, index) => (
              <span
                key={index}
                className="text-xs px-2 py-1 bg-purple-100 text-purple-700 rounded-full"
              >
                {agent}
              </span>
            ))}
            {project.agents.length > 3 && (
              <span className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded-full">
                +{project.agents.length - 3} more
              </span>
            )}
          </div>
        </div>

        {/* Technologies */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-gray-500">Technologies</span>
          </div>
          <div className="flex flex-wrap gap-1">
            {project.technologies.slice(0, 4).map((tech, index) => (
              <span
                key={index}
                className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full"
              >
                {tech}
              </span>
            ))}
            {project.technologies.length > 4 && (
              <span className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded-full">
                +{project.technologies.length - 4}
              </span>
            )}
          </div>
        </div>

        {/* Repository */}
        {project.repository && (
          <div className="flex items-center space-x-2">
            <GitBranch className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-600 truncate">{project.repository}</span>
          </div>
        )}

        {/* Actions */}
        <div className="flex space-x-2 pt-2">
          <Button
            variant="primary"
            size="sm"
            className="flex-1"
            onClick={() => onOpen?.(project.id)}
          >
            Open Project
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onEdit?.(project.id)}
          >
            Edit
          </Button>
        </div>

        {/* Last Activity */}
        <p className="text-xs text-gray-400 text-center">
          Last activity: {project.lastActivity.toLocaleDateString()}
        </p>
      </div>
    </GlassCard>
  );
};

export const ProjectGrid: React.FC<{
  projects: Project[];
  onOpen?: (projectId: string) => void;
  onEdit?: (projectId: string) => void;
  onDelete?: (projectId: string) => void;
  className?: string;
}> = ({ projects, onOpen, onEdit, onDelete: _onDelete, className = '' }) => {
  return (
    <div className={clsx('grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6', className)}>
      {projects.map((project, index) => (
        <motion.div
          key={project.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <ProjectCard
            project={project}
            onOpen={onOpen}
            onEdit={onEdit}
            onDelete={_onDelete}
          />
        </motion.div>
      ))}
    </div>
  );
}; 