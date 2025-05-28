'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { usePathname } from 'next/navigation';
import { 
  Home, 
  Bot, 
  FileText, 
  Settings, 
  ChevronLeft,
  ChevronRight,
  MessageSquare,
  X
} from 'lucide-react';
import { clsx } from 'clsx';
import Link from 'next/link';

interface SidebarProps {
  collapsed?: boolean;
  onToggle?: () => void;
  isMobile?: boolean;
  isOpen?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  collapsed = false, 
  onToggle,
  isMobile = false,
  isOpen = true
}) => {
  const pathname = usePathname();

  const menuItems = [
    { name: 'Dashboard', icon: Home, href: '/' },
    { name: 'Chat', icon: MessageSquare, href: '/chat' },
    { name: 'Agents', icon: Bot, href: '/agents' },
    { name: 'Documents', icon: FileText, href: '/documents' },
    { name: 'Settings', icon: Settings, href: '/settings' }
  ];

  const sidebarVariants = {
    open: {
      x: 0,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 30,
      }
    },
    closed: {
      x: isMobile ? -280 : 0,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 30,
      }
    }
  };

  return (
    <motion.aside
      variants={sidebarVariants}
      initial={isMobile ? "closed" : "open"}
      animate={isOpen ? "open" : "closed"}
      className={clsx(
        // Base styles
        'bg-white/80 backdrop-blur-xl border-r border-gray-200/50',
        'shadow-lg transition-all duration-300',
        
        // Mobile styles
        isMobile && [
          'fixed left-0 top-0 h-full w-72 z-50',
          'safe-top safe-bottom'
        ],
        
        // Desktop styles
        !isMobile && [
          'relative h-full',
          collapsed ? 'w-16' : 'w-64'
        ]
      )}
    >
      {/* Mobile Header */}
      {isMobile && (
        <div className="flex items-center justify-between p-4 border-b border-gray-200/50">
          <h2 className="text-lg font-semibold text-gray-900">Menu</h2>
          <button
            onClick={onToggle}
            className={clsx(
              'p-2 rounded-xl transition-all duration-200',
              'hover:bg-gray-100 active:scale-95'
            )}
          >
            <X className="w-5 h-5 text-gray-600" />
          </button>
        </div>
      )}

      {/* Desktop Toggle Button */}
      {!isMobile && (
        <button
          onClick={onToggle}
          className={clsx(
            'absolute -right-3 top-6 w-6 h-6',
            'bg-gradient-to-br from-system-blue to-system-indigo',
            'rounded-full flex items-center justify-center text-white',
            'shadow-md hover:shadow-lg transition-all duration-200',
            'transform hover:scale-105 active:scale-95',
            'z-10'
          )}
        >
          {collapsed ? (
            <ChevronRight className="w-3 h-3" />
          ) : (
            <ChevronLeft className="w-3 h-3" />
          )}
        </button>
      )}

      {/* Navigation */}
      <nav className={clsx(
        'p-4 space-y-2',
        isMobile && 'pt-0' // Remove top padding on mobile since we have header
      )}>
        {menuItems.map((item, index) => {
          const isActive = pathname === item.href;
          
          return (
            <motion.div
              key={item.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Link
                href={item.href}
                onClick={isMobile ? onToggle : undefined} // Close mobile sidebar on navigation
                className={clsx(
                  'flex items-center space-x-3 px-3 py-3 rounded-xl',
                  'transition-all duration-200 group relative',
                  'hover:bg-gray-100 active:scale-95',
                  isActive && 'bg-system-blue/10 text-system-blue',
                  !isActive && 'text-gray-600 hover:text-system-blue'
                )}
              >
                <item.icon className={clsx(
                  'w-5 h-5 flex-shrink-0 transition-colors duration-200',
                  isActive ? 'text-system-blue' : 'text-gray-600 group-hover:text-system-blue'
                )} />
                
                {(!collapsed || isMobile) && (
                  <motion.span 
                    className={clsx(
                      'font-medium transition-colors duration-200',
                      isActive ? 'text-system-blue' : 'text-gray-700 group-hover:text-system-blue'
                    )}
                    initial={collapsed && !isMobile ? { opacity: 0, width: 0 } : { opacity: 1, width: 'auto' }}
                    animate={collapsed && !isMobile ? { opacity: 0, width: 0 } : { opacity: 1, width: 'auto' }}
                    transition={{ duration: 0.2 }}
                  >
                    {item.name}
                  </motion.span>
                )}

                {/* Active indicator */}
                {isActive && (
                  <motion.div
                    layoutId="sidebarActiveIndicator"
                    className="absolute inset-0 bg-system-blue/10 rounded-xl -z-10"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
              </Link>

              {/* Tooltip for collapsed state */}
              {collapsed && !isMobile && (
                <div className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-50">
                  {item.name}
                </div>
              )}
            </motion.div>
          );
        })}
      </nav>

      {/* Footer for mobile */}
      {isMobile && (
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200/50 bg-white/50">
          <div className="text-xs text-gray-500 text-center">
            Code Hero v1.0
          </div>
        </div>
      )}
    </motion.aside>
  );
}; 