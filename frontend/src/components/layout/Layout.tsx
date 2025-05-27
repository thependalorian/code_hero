'use client';

import React from 'react';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { Footer } from './Footer';
import { clsx } from 'clsx';

interface LayoutProps {
  children: React.ReactNode;
  className?: string;
  showSidebar?: boolean;
  sidebarCollapsed?: boolean;
  onSidebarToggle?: () => void;
}

export const Layout: React.FC<LayoutProps> = ({
  children,
  className = '',
  showSidebar = true,
  sidebarCollapsed = false,
  onSidebarToggle
}) => {
  return (
    <div className="min-h-screen bg-mesh-light">
      {/* Header */}
      <Header />
      
      <div className="flex flex-1">
        {/* Sidebar */}
        {showSidebar && (
          <Sidebar 
            collapsed={sidebarCollapsed}
            onToggle={onSidebarToggle}
          />
        )}
        
        {/* Main Content */}
        <main className={clsx(
          'flex-1 transition-all duration-300',
          showSidebar && !sidebarCollapsed && 'ml-64',
          showSidebar && sidebarCollapsed && 'ml-16',
          className
        )}>
          <div className="container mx-auto px-6 py-8">
            {children}
          </div>
        </main>
      </div>
      
      {/* Footer */}
      <Footer />
    </div>
  );
}; 