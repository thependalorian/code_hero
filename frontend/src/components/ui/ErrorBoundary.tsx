'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, RefreshCw, Home, Bug, Mail } from 'lucide-react';
import { Button } from './Button';
import { GlassCard } from './GlassCard';
import { clsx } from 'clsx';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({ errorInfo });
    this.props.onError?.(error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <ErrorDisplay
          error={this.state.error}
          errorInfo={this.state.errorInfo}
          onRetry={this.handleRetry}
        />
      );
    }

    return this.props.children;
  }
}

// Error display component
interface ErrorDisplayProps {
  error?: Error;
  errorInfo?: ErrorInfo;
  onRetry?: () => void;
  title?: string;
  description?: string;
  showDetails?: boolean;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  error,
  errorInfo,
  onRetry,
  title = 'Something went wrong',
  description = 'An unexpected error occurred. Please try again or contact support if the problem persists.',
  showDetails = false
}) => {
  const [showErrorDetails, setShowErrorDetails] = React.useState(showDetails);

  return (
    <div className="min-h-[400px] flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-md"
      >
        <GlassCard className="p-8 text-center space-y-6">
          {/* Error Icon */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            className="w-16 h-16 mx-auto bg-gradient-to-br from-system-red to-system-orange rounded-2xl flex items-center justify-center shadow-lg"
          >
            <AlertTriangle className="w-8 h-8 text-white" />
          </motion.div>

          {/* Error Content */}
          <div className="space-y-3">
            <h2 className="text-xl font-semibold text-gray-900">
              {title}
            </h2>
            <p className="text-gray-600 leading-relaxed">
              {description}
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3">
            {onRetry && (
              <Button
                onClick={onRetry}
                className={clsx(
                  'bg-gradient-to-r from-system-blue to-system-indigo',
                  'text-white font-semibold flex items-center justify-center space-x-2',
                  'hover:shadow-lg transform hover:scale-105 transition-all duration-200'
                )}
              >
                <RefreshCw className="w-4 h-4" />
                <span>Try Again</span>
              </Button>
            )}
            
            <Button
              variant="ghost"
              onClick={() => window.location.href = '/'}
              className="text-gray-700 hover:bg-gray-100 flex items-center justify-center space-x-2"
            >
              <Home className="w-4 h-4" />
              <span>Go Home</span>
            </Button>
          </div>

          {/* Error Details Toggle */}
          {(error || errorInfo) && (
            <div className="pt-4 border-t border-gray-200/50">
              <button
                onClick={() => setShowErrorDetails(!showErrorDetails)}
                className="text-sm text-gray-500 hover:text-gray-700 transition-colors duration-200 flex items-center space-x-1 mx-auto"
              >
                <Bug className="w-4 h-4" />
                <span>{showErrorDetails ? 'Hide' : 'Show'} Error Details</span>
              </button>

              {showErrorDetails && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="mt-4 p-4 bg-gray-50 rounded-lg text-left"
                >
                  <div className="space-y-2 text-xs font-mono text-gray-700">
                    {error && (
                      <div>
                        <strong>Error:</strong> {error.message}
                      </div>
                    )}
                    {error?.stack && (
                      <div>
                        <strong>Stack:</strong>
                        <pre className="mt-1 whitespace-pre-wrap break-all">
                          {error.stack}
                        </pre>
                      </div>
                    )}
                    {errorInfo?.componentStack && (
                      <div>
                        <strong>Component Stack:</strong>
                        <pre className="mt-1 whitespace-pre-wrap break-all">
                          {errorInfo.componentStack}
                        </pre>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </div>
          )}

          {/* Support Contact */}
          <div className="pt-4 border-t border-gray-200/50">
            <p className="text-xs text-gray-500 mb-2">
              Need help? Contact our support team
            </p>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => window.location.href = 'mailto:support@codehero.dev'}
              className="text-system-blue hover:bg-system-blue/10 flex items-center space-x-1"
            >
              <Mail className="w-3 h-3" />
              <span>support@codehero.dev</span>
            </Button>
          </div>
        </GlassCard>
      </motion.div>
    </div>
  );
};

// Simple error state component for inline errors
interface ErrorStateProps {
  title?: string;
  description?: string;
  onRetry?: () => void;
  className?: string;
  compact?: boolean;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Error',
  description = 'Something went wrong',
  onRetry,
  className = '',
  compact = false
}) => {
  return (
    <div className={clsx(
      'flex flex-col items-center justify-center text-center',
      compact ? 'py-8 space-y-3' : 'py-12 space-y-4',
      className
    )}>
      <div className={clsx(
        'bg-gradient-to-br from-system-red to-system-orange rounded-xl flex items-center justify-center text-white shadow-lg',
        compact ? 'w-12 h-12' : 'w-16 h-16'
      )}>
        <AlertTriangle className={compact ? 'w-6 h-6' : 'w-8 h-8'} />
      </div>
      
      <div className="space-y-2">
        <h3 className={clsx(
          'font-semibold text-gray-900',
          compact ? 'text-lg' : 'text-xl'
        )}>
          {title}
        </h3>
        <p className={clsx(
          'text-gray-600',
          compact ? 'text-sm' : 'text-base'
        )}>
          {description}
        </p>
      </div>
      
      {onRetry && (
        <Button
          onClick={onRetry}
          size={compact ? 'sm' : 'md'}
          className="bg-gradient-to-r from-system-blue to-system-indigo text-white flex items-center space-x-2"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Try Again</span>
        </Button>
      )}
    </div>
  );
}; 