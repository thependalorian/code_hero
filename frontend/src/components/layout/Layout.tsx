'use client';

import React, { useState, useEffect } from 'react';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { Footer } from './Footer';
import { clsx } from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';

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
  const [isMobile, setIsMobile] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Detect mobile screen size
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024); // lg breakpoint
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Handle sidebar toggle for mobile
  const handleSidebarToggle = () => {
    if (isMobile) {
      setSidebarOpen(!sidebarOpen);
    } else {
      onSidebarToggle?.();
    }
  };

  // Close mobile sidebar when clicking outside
  const handleBackdropClick = () => {
    if (isMobile && sidebarOpen) {
      setSidebarOpen(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100 text-rendering-optimized">
      {/* Header */}
      <Header 
        onSidebarToggle={handleSidebarToggle}
      />
      
      <div className="flex relative">
        {/* Mobile Backdrop */}
        <AnimatePresence>
          {isMobile && sidebarOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 lg:hidden"
              onClick={handleBackdropClick}
            />
          )}
        </AnimatePresence>

        {/* Sidebar */}
        {showSidebar && (
          <motion.div
            className={clsx(
              'fixed lg:relative z-50 lg:z-auto',
              'h-[calc(100vh-4rem)] lg:h-auto', // Account for header height
              'top-16 lg:top-0', // Position below header on mobile
              isMobile ? 'left-0' : '',
              !isMobile && sidebarCollapsed && 'lg:w-16',
              !isMobile && !sidebarCollapsed && 'lg:w-64'
            )}
            initial={false}
            animate={{
              x: isMobile ? (sidebarOpen ? 0 : -280) : 0,
            }}
            transition={{
              type: "spring",
              stiffness: 300,
              damping: 30,
            }}
          >
            <Sidebar 
              collapsed={isMobile ? false : sidebarCollapsed}
              onToggle={handleSidebarToggle}
              isMobile={isMobile}
              isOpen={isMobile ? sidebarOpen : true}
            />
          </motion.div>
        )}
        
        {/* Main Content */}
        <motion.main 
          className={clsx(
            'flex-1 min-h-[calc(100vh-4rem)]', // Account for header
            'transition-all duration-300 ease-out',
            // Desktop spacing
            !isMobile && showSidebar && !sidebarCollapsed && 'lg:ml-64',
            !isMobile && showSidebar && sidebarCollapsed && 'lg:ml-16',
            // Mobile spacing (no margin as sidebar is overlay)
            isMobile && 'ml-0',
            className
          )}
          layout
        >
          {/* Content Container with Apple-inspired spacing */}
          <div className={clsx(
            'container mx-auto',
            'px-4 sm:px-6 lg:px-8 xl:px-12', // Responsive horizontal padding
            'py-6 sm:py-8 lg:py-12', // Responsive vertical padding
            'max-w-none lg:max-w-7xl', // Constrain max width on large screens
            'space-y-6 sm:space-y-8 lg:space-y-12' // Responsive vertical spacing
          )}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ 
                duration: 0.4, 
                ease: [0.25, 0.46, 0.45, 0.94] // Apple's easing curve
              }}
            >
              {children}
            </motion.div>
          </div>
        </motion.main>
      </div>
      
      {/* Footer */}
      <Footer />
      
      {/* Safe area padding for mobile devices */}
      <div className="safe-bottom" />
    </div>
  );
}; 