'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';
import { clsx } from 'clsx';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'accent' | 'ghost' | 'glass';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  isLoading?: boolean;
  children: React.ReactNode;
  className?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  isLoading = false,
  children,
  className = '',
  leftIcon,
  rightIcon,
  disabled,
  ...props
}) => {
  const baseClasses = clsx(
    'relative overflow-hidden font-medium rounded-xl transition-all duration-300',
    'transform hover:scale-105 active:scale-95 focus:outline-none focus:ring-2',
    'focus:ring-blue-500/50 focus:ring-offset-2 disabled:opacity-50',
    'disabled:cursor-not-allowed disabled:transform-none',
    'flex items-center justify-center gap-2'
  );
  
  const variantClasses = {
    primary: 'bg-gradient-primary text-white shadow-lg hover:shadow-xl',
    secondary: 'bg-white/80 backdrop-blur-sm text-gray-900 border border-white/20 hover:bg-white shadow-md hover:shadow-lg',
    accent: 'bg-gradient-accent text-white shadow-lg hover:shadow-xl',
    ghost: 'bg-transparent text-gray-700 hover:bg-white/50 hover:backdrop-blur-sm',
    glass: 'glass text-gray-900 hover:bg-white/20'
  };
  
  const sizeClasses = {
    sm: 'px-4 py-2 text-sm min-h-[36px]',
    md: 'px-6 py-3 text-base min-h-[44px]',
    lg: 'px-8 py-4 text-lg min-h-[52px]',
    xl: 'px-10 py-5 text-xl min-h-[60px]'
  };

  const shimmerVariants = {
    initial: { x: '-100%' },
    hover: { x: '100%' }
  };

  return (
    <motion.button
      whileHover={{ scale: disabled || isLoading ? 1 : 1.02 }}
      whileTap={{ scale: disabled || isLoading ? 1 : 0.98 }}
      className={clsx(baseClasses, variantClasses[variant], sizeClasses[size], className)}
      disabled={disabled || isLoading}
      {...props}
    >
      {/* Shimmer effect for gradient buttons */}
      {(variant === 'primary' || variant === 'accent') && (
        <motion.div
          className="absolute inset-0 bg-shimmer"
          variants={shimmerVariants}
          initial="initial"
          whileHover="hover"
          transition={{ duration: 0.5 }}
        />
      )}
      
      {/* Content */}
      <span className="relative z-10 flex items-center gap-2">
        {isLoading ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          leftIcon && <span className="flex-shrink-0">{leftIcon}</span>
        )}
        
        <span className={clsx(isLoading && 'opacity-70')}>
          {children}
        </span>
        
        {!isLoading && rightIcon && (
          <span className="flex-shrink-0">{rightIcon}</span>
        )}
      </span>
    </motion.button>
  );
}; 