'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Code, Heart, Github, Twitter, Linkedin } from 'lucide-react';
import { clsx } from 'clsx';

interface FooterProps {
  className?: string;
}

export const Footer: React.FC<FooterProps> = ({ className = '' }) => {
  const currentYear = new Date().getFullYear();

  const socialLinks = [
    { name: 'GitHub', icon: Github, href: 'https://github.com/code-hero' },
    { name: 'Twitter', icon: Twitter, href: 'https://twitter.com/code-hero' },
    { name: 'LinkedIn', icon: Linkedin, href: 'https://linkedin.com/company/code-hero' }
  ];

  const footerLinks = [
    {
      title: 'Product',
      links: [
        { name: 'Features', href: '/features' },
        { name: 'Pricing', href: '/pricing' },
        { name: 'Documentation', href: '/docs' },
        { name: 'API Reference', href: '/api' }
      ]
    },
    {
      title: 'Company',
      links: [
        { name: 'About', href: '/about' },
        { name: 'Blog', href: '/blog' },
        { name: 'Careers', href: '/careers' },
        { name: 'Contact', href: '/contact' }
      ]
    },
    {
      title: 'Support',
      links: [
        { name: 'Help Center', href: '/help' },
        { name: 'Community', href: '/community' },
        { name: 'Status', href: '/status' },
        { name: 'Security', href: '/security' }
      ]
    },
    {
      title: 'Legal',
      links: [
        { name: 'Privacy', href: '/privacy' },
        { name: 'Terms', href: '/terms' },
        { name: 'Cookies', href: '/cookies' },
        { name: 'Licenses', href: '/licenses' }
      ]
    }
  ];

  return (
    <footer className={clsx(
      'mt-auto border-t border-gray-200/50',
      'bg-white/80 backdrop-blur-xl',
      'safe-bottom',
      className
    )}>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Footer Content */}
        <div className="py-12 lg:py-16">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-8 lg:gap-12">
            {/* Brand Section */}
            <div className="lg:col-span-2 space-y-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="flex items-center space-x-3"
              >
                <div className="w-10 h-10 bg-gradient-to-br from-system-blue to-system-indigo rounded-xl flex items-center justify-center shadow-lg">
                  <Code className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900">Code Hero</h3>
              </motion.div>
              
              <p className="text-gray-600 leading-relaxed max-w-md">
                Empowering developers with AI-powered tools to build amazing applications faster and more efficiently.
              </p>
              
              {/* Social Links */}
              <div className="flex items-center space-x-4">
                {socialLinks.map((social) => (
                  <motion.a
                    key={social.name}
                    href={social.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                    className={clsx(
                      'w-10 h-10 rounded-xl bg-gray-100 hover:bg-gray-200',
                      'flex items-center justify-center text-gray-600 hover:text-system-blue',
                      'transition-all duration-200'
                    )}
                    aria-label={social.name}
                  >
                    <social.icon className="w-5 h-5" />
                  </motion.a>
                ))}
              </div>
            </div>

            {/* Footer Links */}
            {footerLinks.map((section, index) => (
              <motion.div
                key={section.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="space-y-4"
              >
                <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wider">
                  {section.title}
                </h4>
                <ul className="space-y-3">
                  {section.links.map((link) => (
                    <li key={link.name}>
                      <a
                        href={link.href}
                        className={clsx(
                          'text-gray-600 hover:text-system-blue',
                          'transition-colors duration-200',
                          'text-sm leading-relaxed'
                        )}
                      >
                        {link.name}
                      </a>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Bottom Bar */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className={clsx(
            'py-6 border-t border-gray-200/50',
            'flex flex-col sm:flex-row items-center justify-between',
            'space-y-4 sm:space-y-0'
          )}
        >
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <span>© {currentYear} Code Hero. Made with</span>
            <Heart className="w-4 h-4 text-system-red fill-current" />
            <span>for developers.</span>
          </div>
          
          <div className="flex items-center space-x-6 text-sm text-gray-600">
            <span>Version 1.0.0</span>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-system-green rounded-full animate-pulse-gentle" />
              <span>All systems operational</span>
            </div>
          </div>
        </motion.div>
      </div>
    </footer>
  );
}; 