'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Loader2, Code } from 'lucide-react';
import { clsx } from 'clsx';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  variant?: 'default' | 'dots' | 'pulse' | 'brand';
  className?: string;
  text?: string;
  fullScreen?: boolean;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  variant = 'default',
  className = '',
  text,
  fullScreen = false
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
    xl: 'text-xl'
  };

  const renderSpinner = () => {
    switch (variant) {
      case 'dots':
        return (
          <div className="flex items-center space-x-1">
            {[0, 1, 2].map((index) => (
              <motion.div
                key={index}
                className={clsx(
                  'bg-system-blue rounded-full',
                  size === 'sm' && 'w-1 h-1',
                  size === 'md' && 'w-2 h-2',
                  size === 'lg' && 'w-3 h-3',
                  size === 'xl' && 'w-4 h-4'
                )}
                animate={{
                  scale: [1, 1.5, 1],
                  opacity: [0.5, 1, 0.5]
                }}
                transition={{
                  duration: 1.2,
                  repeat: Infinity,
                  delay: index * 0.2,
                  ease: "easeInOut"
                }}
              />
            ))}
          </div>
        );

      case 'pulse':
        return (
          <motion.div
            className={clsx(
              'bg-gradient-to-r from-system-blue to-system-indigo rounded-full',
              sizeClasses[size]
            )}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.7, 1, 0.7]
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        );

      case 'brand':
        return (
          <motion.div
            className={clsx(
              'bg-gradient-to-br from-system-blue to-system-indigo rounded-xl',
              'flex items-center justify-center text-white shadow-lg',
              size === 'sm' && 'w-8 h-8',
              size === 'md' && 'w-12 h-12',
              size === 'lg' && 'w-16 h-16',
              size === 'xl' && 'w-20 h-20'
            )}
            animate={{
              rotate: [0, 360]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "linear"
            }}
          >
            <Code className={clsx(
              size === 'sm' && 'w-4 h-4',
              size === 'md' && 'w-6 h-6',
              size === 'lg' && 'w-8 h-8',
              size === 'xl' && 'w-10 h-10'
            )} />
          </motion.div>
        );

      default:
        return (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: "linear"
            }}
          >
            <Loader2 className={clsx(sizeClasses[size], 'text-system-blue')} />
          </motion.div>
        );
    }
  };

  const content = (
    <div className={clsx(
      'flex flex-col items-center justify-center space-y-3',
      className
    )}>
      {renderSpinner()}
      {text && (
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className={clsx(
            'text-gray-600 font-medium text-center',
            textSizeClasses[size]
          )}
        >
          {text}
        </motion.p>
      )}
    </div>
  );

  if (fullScreen) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="fixed inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-50"
      >
        {content}
      </motion.div>
    );
  }

  return content;
};

// Skeleton loading component for content placeholders
interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  animation?: boolean;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  variant = 'rectangular',
  animation = true
}) => {
  const baseClasses = clsx(
    'bg-gray-200',
    animation && 'animate-pulse',
    variant === 'text' && 'h-4 rounded',
    variant === 'circular' && 'rounded-full',
    variant === 'rectangular' && 'rounded-lg',
    className
  );

  return <div className={baseClasses} />;
};

// Loading overlay for components
interface LoadingOverlayProps {
  isLoading: boolean;
  children: React.ReactNode;
  text?: string;
  variant?: LoadingSpinnerProps['variant'];
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  isLoading,
  children,
  text = 'Loading...',
  variant = 'default'
}) => {
  return (
    <div className="relative">
      {children}
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-10 rounded-lg"
        >
          <LoadingSpinner variant={variant} text={text} />
        </motion.div>
      )}
    </div>
  );
}; 