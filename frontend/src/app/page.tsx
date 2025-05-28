'use client';

import React from 'react';
import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-base-200">
      <div className="hero min-h-screen">
        <div className="hero-content text-center">
          <div className="max-w-md">
            <h1 className="text-5xl font-bold">🦸 Code Hero</h1>
            <p className="py-6 text-lg">
              Your AI-powered development assistant with hierarchical multi-agent system
            </p>
            <div className="flex gap-4 justify-center">
              <Link href="/chat" className="btn btn-primary btn-lg">
                Start Chatting
              </Link>
              <Link href="/agents" className="btn btn-secondary btn-lg">
                View Agents
              </Link>
            </div>
            
            <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="card bg-base-100 shadow-xl">
                <div className="card-body">
                  <h2 className="card-title">💬 Chat Interface</h2>
                  <p>Interact with our AI agents to build amazing applications</p>
                  <div className="card-actions justify-end">
                    <Link href="/chat" className="btn btn-primary btn-sm">
                      Go to Chat
                    </Link>
                  </div>
                </div>
              </div>
              
              <div className="card bg-base-100 shadow-xl">
                <div className="card-body">
                  <h2 className="card-title">🤖 Agent System</h2>
                  <p>Monitor and manage the hierarchical multi-agent system</p>
                  <div className="card-actions justify-end">
                    <Link href="/agents" className="btn btn-secondary btn-sm">
                      View Agents
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
