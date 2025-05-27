'use client';

import React from 'react';
import { motion } from 'framer-motion';

export const Footer: React.FC = () => {
  return (
    <motion.footer
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass border-t border-white/20 py-6 mt-auto"
    >
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between">
          <p className="text-gray-600 text-sm">
            © 2025 Code Hero. Built with ❤️ using Next.js and FastAPI.
          </p>
          <div className="flex items-center space-x-4">
            <span className="text-xs text-gray-500">
              Powered by AI Agents
            </span>
          </div>
        </div>
      </div>
    </motion.footer>
  );
}; 