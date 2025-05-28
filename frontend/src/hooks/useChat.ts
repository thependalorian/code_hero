/**
 * React hook for chat functionality
 * Manages chat state and API communication
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { apiClient, type Message, type ChatResponse } from '@/utils/api';

interface UseChatOptions {
  conversationId?: string;
  autoLoadHistory?: boolean;
}

interface UseChatReturn {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  conversationId: string | null;
  sendMessage: (content: string) => Promise<void>;
  loadHistory: () => Promise<void>;
  clearError: () => void;
  clearMessages: () => void;
}

export function useChat(options: UseChatOptions = {}): UseChatReturn {
  const { conversationId: initialConversationId, autoLoadHistory = true } = options;
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(
    initialConversationId || null
  );

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setConversationId(null);
  }, []);

  const loadHistory = useCallback(async () => {
    if (!conversationId) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.getChatHistory(conversationId);
      setMessages(response.messages || []);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load chat history';
      setError(errorMessage);
      console.error('Failed to load chat history:', err);
    } finally {
      setIsLoading(false);
    }
  }, [conversationId]);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return;

    setIsLoading(true);
    setError(null);

    // Add user message immediately for better UX
    const userMessage: Message = {
      role: 'user',
      content: content.trim(),
      timestamp: new Date().toISOString(),
      metadata: { source: 'frontend' }
    };

    setMessages(prev => [...prev, userMessage]);

    try {
      const response: ChatResponse = await apiClient.sendMessage(
        content.trim(),
        conversationId || undefined
      );

      // Update conversation ID if this is a new conversation
      if (!conversationId && response.conversation_id) {
        setConversationId(response.conversation_id);
      }

      // Update messages with the full conversation from backend
      if (response.messages && response.messages.length > 0) {
        setMessages(response.messages);
      } else if (response.response) {
        // Fallback: add assistant message if messages array is not provided
        const assistantMessage: Message = {
          role: 'assistant',
          content: response.response,
          timestamp: new Date().toISOString(),
          metadata: { 
            agent: response.active_agent,
            status: response.status 
          }
        };
        setMessages(prev => [...prev, assistantMessage]);
      }

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message';
      setError(errorMessage);
      console.error('Failed to send message:', err);

      // Add error message to chat
      const errorMsg: Message = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${errorMessage}. Please try again.`,
        timestamp: new Date().toISOString(),
        metadata: { error: true }
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [conversationId]);

  // Auto-load history on mount if conversationId is provided
  const hasLoadedRef = useRef(false);
  
  useEffect(() => {
    if (autoLoadHistory && conversationId && !hasLoadedRef.current) {
      hasLoadedRef.current = true;
      loadHistory();
    }
  }, [autoLoadHistory, conversationId, loadHistory]);

  return {
    messages,
    isLoading,
    error,
    conversationId,
    sendMessage,
    loadHistory,
    clearError,
    clearMessages,
  };
} 