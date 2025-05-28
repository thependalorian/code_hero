'use client';

import React, { forwardRef } from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';

interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  variant?: 'default' | 'glass' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  error?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  isLoading?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(({
  variant = 'glass',
  size = 'md',
  label,
  error,
  leftIcon,
  rightIcon,
  isLoading = false,
  className = '',
  ...props
}, ref) => {
  const baseClasses = clsx(
    'w-full rounded-xl transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-blue-500/50',
    'placeholder-gray-500 text-gray-900'
  );

  const variantClasses = {
    default: 'bg-white border border-gray-200 hover:border-gray-300',
    glass: 'glass border border-white/20 hover:border-white/30',
    outline: 'bg-transparent border-2 border-gray-200 hover:border-blue-300'
  };

  const sizeClasses = {
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-3 text-base',
    lg: 'px-5 py-4 text-lg'
  };

  const iconSizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6'
  };

  return (
    <div className="space-y-2">
      {label && (
        <label className="block text-sm font-medium text-gray-700">
          {label}
        </label>
      )}
      
      <div className="relative">
        {leftIcon && (
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
            <div className={iconSizes[size]}>
              {leftIcon}
            </div>
          </div>
        )}
        
        <input
          ref={ref}
          className={clsx(
            baseClasses,
            variantClasses[variant],
            sizeClasses[size],
            leftIcon && 'pl-10',
            rightIcon && 'pr-10',
            error && 'border-red-300 focus:ring-red-500/50',
            className
          )}
          {...props}
        />
        
        {rightIcon && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400">
            <div className={iconSizes[size]}>
              {rightIcon}
            </div>
          </div>
        )}
        
        {isLoading && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        )}
      </div>
      
      {error && (
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-sm text-red-600"
        >
          {error}
        </motion.p>
      )}
    </div>
  );
});

Input.displayName = 'Input'; 