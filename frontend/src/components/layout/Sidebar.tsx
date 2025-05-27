'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  Home, 
  FolderOpen, 
  Bot, 
  FileText, 
  Settings, 
  ChevronLeft,
  ChevronRight 
} from 'lucide-react';
import { clsx } from 'clsx';

interface SidebarProps {
  collapsed?: boolean;
  onToggle?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  collapsed = false, 
  onToggle 
}) => {
  const menuItems = [
    { name: 'Dashboard', icon: Home, href: '/', active: true },
    { name: 'Projects', icon: FolderOpen, href: '/projects', active: false },
    { name: 'Agents', icon: Bot, href: '/agents', active: false },
    { name: 'Documents', icon: FileText, href: '/documents', active: false },
    { name: 'Settings', icon: Settings, href: '/settings', active: false }
  ];

  return (
    <motion.aside
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      className={clsx(
        'fixed left-0 top-16 h-[calc(100vh-4rem)] glass border-r border-white/20',
        'transition-all duration-300 z-40',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Toggle Button */}
      <button
        onClick={onToggle}
        className="absolute -right-3 top-6 w-6 h-6 bg-gradient-primary rounded-full flex items-center justify-center text-white shadow-lg hover:shadow-xl transition-all duration-300"
      >
        {collapsed ? (
          <ChevronRight className="w-3 h-3" />
        ) : (
          <ChevronLeft className="w-3 h-3" />
        )}
      </button>

      {/* Navigation */}
      <nav className="p-4 space-y-2">
        {menuItems.map((item, index) => (
          <motion.a
            key={item.name}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            href={item.href}
            className={clsx(
              'flex items-center space-x-3 px-3 py-3 rounded-xl transition-all duration-200',
              'hover:bg-white/20 group',
              item.active && 'bg-blue-100 text-blue-600'
            )}
          >
            <item.icon className={clsx(
              'w-5 h-5 flex-shrink-0',
              item.active ? 'text-blue-600' : 'text-gray-600 group-hover:text-blue-600'
            )} />
            
            {!collapsed && (
              <span className={clsx(
                'font-medium transition-colors duration-200',
                item.active ? 'text-blue-600' : 'text-gray-700 group-hover:text-blue-600'
              )}>
                {item.name}
              </span>
            )}
          </motion.a>
        ))}
      </nav>
    </motion.aside>
  );
}; 