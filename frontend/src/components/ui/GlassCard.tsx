'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  glow?: boolean;
  variant?: 'default' | 'dark' | 'light';
  padding?: 'none' | 'sm' | 'md' | 'lg' | 'xl';
  onClick?: () => void;
  as?: 'div' | 'article' | 'section';
}

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className = '',
  hover = true,
  glow = false,
  variant = 'default',
  padding = 'md',
  onClick,
  as: Component = 'div'
}) => {
  const baseClasses = clsx(
    'rounded-2xl border transition-all duration-300',
    'backdrop-blur-md shadow-xl',
    onClick && 'cursor-pointer'
  );

  const variantClasses = {
    default: 'bg-white/10 border-white/20 hover:shadow-2xl',
    dark: 'bg-black/10 border-white/10 hover:shadow-2xl',
    light: 'bg-white/80 border-white/30 hover:shadow-2xl'
  };

  const paddingClasses = {
    none: '',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
    xl: 'p-10'
  };

  const glowClasses = glow ? 'shadow-glow-md animate-glow' : '';

  const hoverAnimation = hover ? {
    scale: 1.02,
    y: -4,
    transition: { duration: 0.2, ease: "easeInOut" }
  } : {};

  const cardVariants = {
    hidden: { 
      opacity: 0, 
      y: 20,
      scale: 0.95
    },
    visible: { 
      opacity: 1, 
      y: 0,
      scale: 1,
      transition: { 
        duration: 0.4, 
        ease: "easeOut"
      }
    },
    hover: hoverAnimation
  };

  return (
    <motion.div
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      whileHover={hover ? "hover" : undefined}
      className={clsx(
        baseClasses,
        variantClasses[variant],
        paddingClasses[padding],
        glowClasses,
        className
      )}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
    >
      <div className="relative z-10">
        {children}
      </div>
      
      {/* Subtle inner glow */}
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />
      
      {/* Border highlight */}
      <div className="absolute inset-0 rounded-2xl border border-white/10 pointer-events-none" />
    </motion.div>
  );
};

// Specialized card variants
export const StatusCard: React.FC<GlassCardProps & { status: 'active' | 'processing' | 'error' | 'idle' }> = ({
  status,
  children,
  ...props
}) => {
  const statusColors = {
    active: 'shadow-green-500/20 border-green-500/30',
    processing: 'shadow-yellow-500/20 border-yellow-500/30',
    error: 'shadow-red-500/20 border-red-500/30',
    idle: 'shadow-gray-500/20 border-gray-500/30'
  };

  return (
    <GlassCard 
      {...props}
      className={clsx(statusColors[status], props.className)}
      glow={status === 'active'}
    >
      {children}
    </GlassCard>
  );
};

export const MetricCard: React.FC<GlassCardProps & { 
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
}> = ({
  title,
  value,
  subtitle,
  icon,
  trend,
  ...props
}) => {
  const trendColors = {
    up: 'text-green-500',
    down: 'text-red-500',
    neutral: 'text-gray-500'
  };

  return (
    <GlassCard {...props}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mb-1">{value}</p>
          {subtitle && (
            <p className={clsx(
              'text-sm font-medium',
              trend ? trendColors[trend] : 'text-gray-500'
            )}>
              {subtitle}
            </p>
          )}
        </div>
        {icon && (
          <div className="flex-shrink-0 ml-4">
            <div className="w-12 h-12 bg-gradient-primary rounded-xl flex items-center justify-center text-white">
              {icon}
            </div>
          </div>
        )}
      </div>
    </GlassCard>
  );
};