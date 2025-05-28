'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  Bot, 
  User, 
  Mic, 
  MicOff, 
  Copy, 
  RotateCcw, 
  ChevronDown, 
  Loader2, 
  CheckCircle, 
  Clock,
  Code,
  FileText,
  AlertCircle,
  Zap
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { clsx } from 'clsx';
import { ApiClient } from '@/utils/api';
import type { Message } from '@/utils/api';

interface ChatMessage extends Message {
  id: string;
  status?: 'sending' | 'sent' | 'error';
  agent?: string;
}

interface ChatInterfaceProps {
  className?: string;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ className }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  
  // Initialize API client
  const api = new ApiClient();
  
  // Toast actions (simplified)
  const showToast = useCallback((message: string, type: 'success' | 'error' | 'info' = 'success') => {
    console.log(`${type.toUpperCase()}: ${message}`);
  }, []);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Handle scroll to show/hide scroll button
  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      setShowScrollButton(!isNearBottom && messages.length > 0);
    }
  };

  // Auto-resize textarea
  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputValue]);

  useEffect(() => {
    if (!isLoading && !isTyping) {
      scrollToBottom();
    }
  }, [messages, isLoading, isTyping]);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date().toISOString(),
      status: 'sending'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setIsTyping(true);

    try {
      // Update message status to sent
      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessage.id 
            ? { ...msg, status: 'sent' }
            : msg
        )
      );

      // Send to backend
      const response = await api.sendMessage(userMessage.content, conversationId ?? undefined);
      
      // Set conversation ID if not set
      if (!conversationId && response.conversation_id) {
        setConversationId(response.conversation_id);
      }

      // Add assistant response
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.response,
        timestamp: new Date().toISOString(),
        status: 'sent',
        agent: response.active_agent,
        metadata: {
          conversation_id: response.conversation_id,
          status: response.status
        }
      };

      // Simulate typing delay
      setTimeout(() => {
        setIsTyping(false);
        setMessages(prev => [...prev, assistantMessage]);
      }, 1000);

    } catch (error) {
      console.error('Failed to send message:', error);
      
      // Update user message status to error
      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessage.id 
            ? { ...msg, status: 'error' }
            : msg
        )
      );

      // Add error message
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        timestamp: new Date().toISOString(),
        status: 'error',
        metadata: { error: true }
      };

      setTimeout(() => {
        setIsTyping(false);
        setMessages(prev => [...prev, errorMessage]);
      }, 500);

      showToast('Message failed', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const retryMessage = async (messageId: string) => {
    const message = messages.find(msg => msg.id === messageId);
    if (!message || message.role !== 'user') return;

    // Remove the failed message and any subsequent messages
    const messageIndex = messages.findIndex(msg => msg.id === messageId);
    setMessages(prev => prev.slice(0, messageIndex));
    
    // Resend the message
    setInputValue(message.content);
    setTimeout(() => sendMessage(), 100);
  };

  const copyMessage = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      showToast('Message content copied successfully', 'success');
    } catch {
      showToast('Copy failed', 'error');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const toggleRecording = () => {
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = () => {
        // TODO: Convert audio to text using speech recognition API
        showToast('Voice recording completed', 'info');
        
        // For now, just add a placeholder message
        setInputValue('Voice message recorded (speech-to-text not implemented yet)');
        
        // Clean up
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      showToast('Recording started', 'info');

      // Auto-stop after 30 seconds
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
          setIsRecording(false);
        }
      }, 30000);

    } catch (error) {
      console.error('Error accessing microphone:', error);
      showToast('Microphone access denied', 'error');
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    showToast('Recording stopped', 'info');
  };

  const quickActions = [
    { label: 'Help me code', icon: Code, prompt: 'Help me write some code' },
    { label: 'Explain concept', icon: FileText, prompt: 'Explain a programming concept' },
    { label: 'Debug issue', icon: AlertCircle, prompt: 'Help me debug an issue' },
    { label: 'Optimize code', icon: Zap, prompt: 'Help me optimize my code' }
  ];

  const getMessageStatusIcon = (status?: ChatMessage['status']) => {
    switch (status) {
      case 'sending':
        return <Clock className="w-3 h-3 text-gray-400" />;
      case 'sent':
        return <CheckCircle className="w-3 h-3 text-system-green" />;
      case 'error':
        return <AlertCircle className="w-3 h-3 text-system-red" />;
      default:
        return null;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className={clsx('flex flex-col h-full bg-gray-50/95 backdrop-blur-sm rounded-2xl border border-gray-300/50 shadow-lg', className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200/50 bg-white/80">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-system-blue to-system-indigo rounded-xl flex items-center justify-center text-white shadow-lg">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">Code Hero AI</h3>
            <p className="text-sm text-gray-600">
              {conversationId ? `Conversation ${conversationId.slice(0, 8)}...` : 'Ready to help'}
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 text-xs text-gray-600">
            <div className="w-2 h-2 bg-system-green rounded-full animate-pulse" />
            <span>Online</span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div 
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50/30"
      >
        {messages.length === 0 && (
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto bg-gradient-to-br from-system-blue to-system-indigo rounded-2xl flex items-center justify-center text-white shadow-lg mb-4">
              <Bot className="w-8 h-8" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Welcome to Code Hero AI
            </h3>
            <p className="text-gray-700 mb-6 max-w-md mx-auto">
              I&apos;m here to help you with coding, debugging, architecture decisions, and more. 
              What would you like to work on today?
            </p>
            
            {/* Quick Actions */}
            <div className="grid grid-cols-2 gap-3 max-w-md mx-auto">
              {quickActions.map(({ label, icon: Icon, prompt }) => (
                <button
                  key={label}
                  onClick={() => setInputValue(prompt)}
                  className="flex items-center space-x-2 p-3 bg-white rounded-xl border border-gray-300 hover:bg-gray-50 hover:shadow-md transition-all duration-200 text-left"
                >
                  <Icon className="w-4 h-4 text-system-blue" />
                  <span className="text-sm font-medium text-gray-800">{label}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={clsx(
              'flex items-start space-x-3 group',
              message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
            )}
          >
            {/* Avatar */}
            <div className={clsx(
              'w-8 h-8 rounded-xl flex items-center justify-center text-white shadow-lg flex-shrink-0',
              message.role === 'user' 
                ? 'bg-gradient-to-br from-system-green to-system-teal'
                : 'bg-gradient-to-br from-system-blue to-system-indigo'
            )}>
              {message.role === 'user' ? (
                <User className="w-4 h-4" />
              ) : (
                <Bot className="w-4 h-4" />
              )}
            </div>

            {/* Message Content */}
            <div className={clsx(
              'flex-1 max-w-[80%]',
              message.role === 'user' ? 'text-right' : ''
            )}>
              <div className={clsx(
                'inline-block p-4 rounded-2xl shadow-sm',
                message.role === 'user'
                  ? 'bg-gradient-to-br from-system-blue to-system-indigo text-white'
                  : 'bg-white/90 backdrop-blur-sm border border-gray-300/50 text-gray-900 shadow-sm',
                message.status === 'error' && 'border-system-red/50 bg-system-red/10'
              )}>
                <p className="whitespace-pre-wrap leading-relaxed">
                  {message.content}
                </p>
                
                {/* Agent info for assistant messages */}
                {message.role === 'assistant' && message.agent && (
                  <div className="mt-2 pt-2 border-t border-gray-300/50">
                    <span className="text-xs text-gray-600">
                      Handled by: {message.agent}
                    </span>
                  </div>
                )}
              </div>

              {/* Message metadata */}
              <div className={clsx(
                'flex items-center space-x-2 mt-1 text-xs text-gray-600',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}>
                <span>{formatTimestamp(message.timestamp)}</span>
                {getMessageStatusIcon(message.status)}
                
                {/* Message actions */}
                <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => copyMessage(message.content)}
                    className="p-1 hover:bg-gray-200 rounded transition-colors"
                    title="Copy message"
                  >
                    <Copy className="w-3 h-3" />
                  </button>
                  
                  {message.status === 'error' && message.role === 'user' && (
                    <button
                      onClick={() => retryMessage(message.id)}
                      className="p-1 hover:bg-gray-200 rounded transition-colors"
                      title="Retry message"
                    >
                      <RotateCcw className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        ))}

        {/* Typing indicator */}
        <AnimatePresence>
          {isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex items-start space-x-3"
            >
              <div className="w-8 h-8 bg-gradient-to-br from-system-blue to-system-indigo rounded-xl flex items-center justify-center text-white shadow-lg">
                <Bot className="w-4 h-4" />
              </div>
              <div className="bg-white/90 backdrop-blur-sm border border-gray-300/50 rounded-2xl p-4 shadow-sm">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  <span className="text-sm text-gray-700">AI is thinking...</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div ref={messagesEndRef} />
      </div>

      {/* Scroll to bottom button */}
      <AnimatePresence>
        {showScrollButton && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            onClick={scrollToBottom}
            className="absolute bottom-24 right-6 w-10 h-10 bg-white/95 backdrop-blur-sm border border-gray-300/50 rounded-full shadow-lg flex items-center justify-center hover:bg-white transition-all duration-200"
          >
            <ChevronDown className="w-5 h-5 text-gray-700" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Input */}
      <div className="p-4 border-t border-gray-200/50 bg-white/90">
        <div className="flex items-end space-x-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message... (Shift+Enter for new line)"
              disabled={isLoading}
              className={clsx(
                'w-full px-4 py-3 pr-12 rounded-xl border border-gray-300',
                'bg-white resize-none text-gray-900 placeholder-gray-500',
                'focus:outline-none focus:ring-2 focus:ring-system-blue/30 focus:border-system-blue',
                'transition-all duration-200 min-h-[48px] max-h-[120px] shadow-sm',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
              rows={1}
            />
            
            {/* Character count */}
            <div className="absolute bottom-2 right-2 text-xs text-gray-500">
              {inputValue.length}/2000
            </div>
          </div>

          {/* Voice recording button */}
          <Button
            onClick={toggleRecording}
            variant="ghost"
            size="sm"
            className={clsx(
              'w-12 h-12 rounded-xl transition-all duration-200 border border-gray-300',
              isRecording 
                ? 'bg-system-red text-white hover:bg-system-red/90 border-system-red' 
                : 'text-gray-600 hover:text-system-blue hover:bg-system-blue/10 bg-white'
            )}
            title={isRecording ? 'Stop recording' : 'Start voice recording'}
          >
            {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
          </Button>

          {/* Send button */}
          <Button
            onClick={sendMessage}
            disabled={!inputValue.trim() || isLoading}
            className={clsx(
              'w-12 h-12 rounded-xl bg-gradient-to-r from-system-blue to-system-indigo',
              'text-white font-semibold flex items-center justify-center',
              'hover:shadow-lg transform hover:scale-105 transition-all duration-200',
              'disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none'
            )}
            title="Send message (Enter)"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </Button>
        </div>

        {/* Quick suggestions when input is empty */}
        {!inputValue && messages.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {['Help me debug', 'Explain this code', 'Optimize performance', 'Add tests'].map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => setInputValue(suggestion)}
                className="px-3 py-1 bg-white text-gray-700 text-sm rounded-lg border border-gray-300 hover:bg-gray-50 hover:shadow-sm transition-all duration-200"
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface; 