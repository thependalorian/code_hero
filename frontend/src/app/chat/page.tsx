'use client';

import { ChatInterface } from '@/components/chat/ChatInterface';

export default function ChatPage() {
  return (
    <div className="min-h-screen bg-base-200">
      <div className="container mx-auto py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-primary mb-4">
            Code Hero Chat
          </h1>
          <p className="text-lg text-base-content/70">
            Interact with our AI agents to build amazing applications
          </p>
        </div>
        
        <ChatInterface />
      </div>
    </div>
  );
} 