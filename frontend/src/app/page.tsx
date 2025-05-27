'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Code, 
  Bot, 
  Zap, 
  Users, 
  ArrowRight, 
  Play,
  Sparkles,
  Brain,
  Rocket
} from 'lucide-react';
import { Layout } from '@/components/layout/Layout';
import { Button } from '@/components/ui/Button';
import { GlassCard, MetricCard } from '@/components/ui/GlassCard';

export default function HomePage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const features = [
    {
      icon: Brain,
      title: "Multi-Agent Intelligence",
      description: "Coordinate multiple AI agents working together seamlessly on complex tasks with real-time collaboration.",
      gradient: "from-purple-500 to-pink-500"
    },
    {
      icon: Zap,
      title: "Lightning Fast",
      description: "Built with FastAPI and Next.js for blazing fast performance and real-time updates.",
      gradient: "from-blue-500 to-cyan-500"
    },
    {
      icon: Code,
      title: "Code Generation",
      description: "Generate, validate, and optimize code across multiple languages with AI-powered assistance.",
      gradient: "from-green-500 to-emerald-500"
    },
    {
      icon: Rocket,
      title: "Production Ready",
      description: "Deploy to Vercel with confidence. Built for scale with enterprise-grade security.",
      gradient: "from-orange-500 to-red-500"
    }
  ];

  const metrics = [
    { title: "Active Agents", value: "7", subtitle: "+2 this week", trend: "up" as const, icon: Bot },
    { title: "Projects", value: "12", subtitle: "3 completed", trend: "up" as const, icon: Code },
    { title: "Success Rate", value: "98%", subtitle: "Last 30 days", trend: "up" as const, icon: Zap },
    { title: "Response Time", value: "1.2s", subtitle: "Average", trend: "neutral" as const, icon: Users }
  ];

  return (
    <Layout 
      sidebarCollapsed={sidebarCollapsed}
      onSidebarToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
    >
      {/* Hero Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-3xl bg-gradient-mesh p-12 mb-12"
      >
        <div className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-center max-w-4xl mx-auto"
          >
            <div className="flex items-center justify-center mb-6">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center shadow-2xl"
              >
                <Sparkles className="w-8 h-8 text-white" />
              </motion.div>
            </div>
            
            <h1 className="text-6xl font-bold mb-6">
              <span className="gradient-text">Code Hero</span>
              <br />
              <span className="text-white">AI Agent Platform</span>
            </h1>
            
            <p className="text-xl text-white/80 mb-8 max-w-2xl mx-auto leading-relaxed">
              Experience the future of AI collaboration with our beautiful, Imagica.ai inspired platform. 
              Coordinate multiple agents, generate code, and build amazing projects with stunning visual design.
            </p>
            
            <div className="flex items-center justify-center gap-4">
              <Button 
                size="lg" 
                className="bg-white/20 backdrop-blur-sm text-white border border-white/30 hover:bg-white/30"
                leftIcon={<Play className="w-5 h-5" />}
              >
                Start Building
              </Button>
              <Button 
                variant="ghost" 
                size="lg"
                className="text-white hover:bg-white/20"
                rightIcon={<ArrowRight className="w-5 h-5" />}
              >
                View Demo
              </Button>
            </div>
          </motion.div>
        </div>
        
        {/* Floating elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-32 h-32 bg-white/5 rounded-full"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                y: [0, -20, 0],
                opacity: [0.3, 0.6, 0.3],
              }}
              transition={{
                duration: 3 + Math.random() * 2,
                repeat: Infinity,
                delay: Math.random() * 2,
              }}
            />
          ))}
        </div>
      </motion.section>

      {/* Metrics Grid */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12"
      >
        {metrics.map((metric, index) => (
          <motion.div
            key={metric.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 + index * 0.1 }}
          >
            <MetricCard
              title={metric.title}
              value={metric.value}
              subtitle={metric.subtitle}
              trend={metric.trend}
              icon={<metric.icon className="w-6 h-6" />}
              hover={true}
            />
          </motion.div>
        ))}
      </motion.section>

      {/* Features Grid */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="mb-12"
      >
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold gradient-text mb-4">
            Powerful Features
          </h2>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            Everything you need to build, deploy, and scale AI-powered applications with beautiful design.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 + index * 0.1 }}
            >
              <GlassCard hover={true} className="h-full">
                <div className="flex items-start space-x-4">
                  <div className={`w-12 h-12 bg-gradient-to-r ${feature.gradient} rounded-xl flex items-center justify-center shadow-lg`}>
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600 leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* CTA Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <GlassCard className="text-center bg-gradient-to-r from-blue-50 to-purple-50">
          <h2 className="text-3xl font-bold gradient-text mb-4">
            Ready to Get Started?
          </h2>
          <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
            Join thousands of developers building the future with AI agents. 
            Start your journey today with our beautiful, intuitive platform.
          </p>
          <div className="flex items-center justify-center gap-4">
            <Button size="lg" leftIcon={<Rocket className="w-5 h-5" />}>
              Start Free Trial
            </Button>
            <Button variant="ghost" size="lg">
              Contact Sales
            </Button>
          </div>
        </GlassCard>
      </motion.section>
    </Layout>
  );
}
