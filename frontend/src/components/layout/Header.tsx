'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Code, Bell, Menu, X, User, Settings, LogOut } from 'lucide-react';
import { Button } from '../ui/Button';
import { clsx } from 'clsx';

interface HeaderProps {
  className?: string;
  onSidebarToggle?: () => void;
  sidebarCollapsed?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ 
  className = '',
  onSidebarToggle
}) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);

  const navigationItems = [
    { name: 'Dashboard', href: '/', active: true },
    { name: 'Projects', href: '/projects', active: false },
    { name: 'Agents', href: '/agents', active: false },
    { name: 'Documents', href: '/documents', active: false }
  ];

  const profileMenuItems = [
    { name: 'Profile', icon: User, href: '/profile' },
    { name: 'Settings', icon: Settings, href: '/settings' },
    { name: 'Sign Out', icon: LogOut, href: '/logout' }
  ];

  return (
    <header className={clsx(
      'sticky top-0 z-50',
      'bg-white/80 backdrop-blur-xl border-b border-gray-200/50',
      'shadow-sm',
      'safe-top', // Safe area for mobile devices
      className
    )}>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left Side - Logo and Sidebar Toggle */}
          <div className="flex items-center space-x-4">
            {/* Sidebar Toggle Button */}
            {onSidebarToggle && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onSidebarToggle}
                className={clsx(
                  'p-2 rounded-xl transition-all duration-200',
                  'hover:bg-gray-100 active:scale-95',
                  'lg:hidden' // Only show on mobile/tablet
                )}
                aria-label="Toggle sidebar"
              >
                <Menu className="w-5 h-5 text-gray-600" />
              </Button>
            )}

            {/* Logo */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center space-x-3"
            >
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-br from-system-blue to-system-indigo rounded-xl flex items-center justify-center shadow-lg">
                <Code className="w-4 h-4 sm:w-6 sm:h-6 text-white" />
              </div>
              <h1 className="text-lg sm:text-xl lg:text-2xl font-semibold text-gray-900">
                Code Hero
              </h1>
            </motion.div>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center space-x-1">
            {navigationItems.map((item, index) => (
              <motion.a
                key={item.name}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                href={item.href}
                className={clsx(
                  'relative px-4 py-2 rounded-xl transition-all duration-200',
                  'text-sm font-medium',
                  item.active 
                    ? 'text-system-blue bg-system-blue/10' 
                    : 'text-gray-600 hover:text-system-blue hover:bg-gray-100'
                )}
              >
                {item.name}
                {item.active && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-system-blue/10 rounded-xl -z-10"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
              </motion.a>
            ))}
          </nav>

          {/* Right Side Actions */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center space-x-2 sm:space-x-3"
          >
            {/* Notifications */}
            <Button 
              variant="ghost" 
              size="sm" 
              className="relative p-2 rounded-xl hover:bg-gray-100 transition-all duration-200"
            >
              <Bell className="w-5 h-5 text-gray-600" />
              <span className="absolute -top-1 -right-1 w-2 h-2 bg-system-red rounded-full animate-pulse-gentle" />
            </Button>

            {/* Profile Menu */}
            <div className="relative">
              <button
                onClick={() => setIsProfileMenuOpen(!isProfileMenuOpen)}
                className={clsx(
                  'w-8 h-8 sm:w-10 sm:h-10',
                  'bg-gradient-to-br from-system-pink to-system-purple',
                  'rounded-full flex items-center justify-center',
                  'shadow-md hover:shadow-lg transition-all duration-200',
                  'transform hover:scale-105 active:scale-95'
                )}
              >
                <User className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
              </button>

              {/* Profile Dropdown */}
              {isProfileMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95, y: -10 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95, y: -10 }}
                  className={clsx(
                    'absolute right-0 mt-2 w-48',
                    'glass-medium rounded-2xl shadow-xl border border-white/20',
                    'py-2 z-50'
                  )}
                >
                  {profileMenuItems.map((item) => (
                    <a
                      key={item.name}
                      href={item.href}
                      className={clsx(
                        'flex items-center space-x-3 px-4 py-3',
                        'text-gray-700 hover:bg-white/20',
                        'transition-colors duration-200',
                        'text-sm font-medium'
                      )}
                    >
                      <item.icon className="w-4 h-4" />
                      <span>{item.name}</span>
                    </a>
                  ))}
                </motion.div>
              )}
            </div>

            {/* Mobile Menu Button */}
            <Button
              variant="ghost"
              size="sm"
              className="lg:hidden p-2 rounded-xl hover:bg-gray-100 transition-all duration-200"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              {isMobileMenuOpen ? (
                <X className="w-5 h-5 text-gray-600" />
              ) : (
                <Menu className="w-5 h-5 text-gray-600" />
              )}
            </Button>
          </motion.div>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <motion.nav
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden pb-4 border-t border-gray-200/50 mt-4 pt-4"
          >
            <div className="space-y-1">
              {navigationItems.map((item) => (
                <a
                  key={item.name}
                  href={item.href}
                  className={clsx(
                    'block px-4 py-3 rounded-xl transition-all duration-200',
                    'text-sm font-medium',
                    item.active
                      ? 'bg-system-blue/10 text-system-blue'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-system-blue'
                  )}
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  {item.name}
                </a>
              ))}
            </div>
          </motion.nav>
        )}
      </div>
    </header>
  );
}; 