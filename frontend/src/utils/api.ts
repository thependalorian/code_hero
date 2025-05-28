/**
 * API Client for Code Hero Backend
 * Handles all communication with the FastAPI backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ─────────────────────────────────────────────────────────────────────────────
// TYPE DEFINITIONS (matching backend state.py)
// ─────────────────────────────────────────────────────────────────────────────

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  messages: Message[];
  status: string;
  active_agent: string;
}

export interface ChatHistoryResponse {
  conversation_id: string;
  messages: Message[];
  status: string;
  active_agent: string;
}

export interface HealthResponse {
  status: string;
  services: {
    logger: {
      initialized: boolean;
      level: string;
      has_file_handler: boolean;
      handler_count: number;
    };
    state_manager: {
      initialized: boolean;
      project_count: number;
      chat_count: number;
      agent_count: number;
      graph_count: number;
    };
    supervisor: {
      initialized: boolean;
      status: string | null;
      active_workflows: number;
      state_manager_connected: boolean;
      logger_connected: boolean;
    };
  };
  config: {
    api: boolean;
    database: boolean;
    llm_registry: boolean;
    logging: boolean;
  };
  environment: string;
  timestamp: string;
}

export interface MultiAgentRequest {
  task_description: string;
  project_id?: string;
}

export interface MultiAgentResponse {
  success: boolean;
  project_id: string;
  task_description: string;
  result: unknown;
  timestamp: string;
  error?: string;
}

// Agent-related types (matching backend AgentInfoExtended)
export interface AgentPerformance {
  tasks_completed: number;
  success_rate: number;
  avg_response_time: string;
  uptime: string;
}

export interface Agent {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'processing' | 'idle' | 'error';
  description: string;
  capabilities: string[];
  tools: string[];
  team?: string;
  performance: AgentPerformance;
  current_task?: string;
  last_active: string;
}

export interface AgentInteractionRequest {
  message: string;
}

export interface AgentInteractionResponse {
  success: boolean;
  agent_id: string;
  task_id: string;
  response: string;
  duration: string;
  timestamp: string;
}

export interface AgentHistoryResponse {
  agent_id: string;
  total_tasks: number;
  recent_tasks: Array<{
    task_id: string;
    description: string;
    success: boolean;
    duration: number;
    timestamp: string;
  }>;
  limit: number;
}

export interface AgentStatistics {
  total_agents: number;
  active_agents: number;
  total_tasks_completed: number;
  average_success_rate: number;
  average_response_time: number;
  teams: Array<{
    name: string;
    agent_count: number;
    success_rate: number;
  }>;
}

// Document-related types
export interface DocumentUploadResponse {
  success: boolean;
  uploaded_files: Array<{
    id: string;
    filename: string;
    size: number;
    status: string;
  }>;
  failed_files: Array<{
    filename: string;
    error: string;
  }>;
  total_uploaded: number;
  total_failed: number;
  project_id?: string;
  timestamp: string;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  size: number;
  content_type: string;
  format: string;
  processing_status: string;
  upload_timestamp: string;
  project_id?: string;
  description?: string;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total: number;
  limit: number;
  offset: number;
  project_id?: string;
}

export interface DocumentAnalysisResponse {
  success: boolean;
  document_id: string;
  task_id: string;
  analysis_type: string;
  result: string;
  timestamp: string;
}

export interface TRDConversionResponse {
  success: boolean;
  document_id: string;
  trd_id: string;
  task_id: string;
  target_format: string;
  result: string;
  stakeholders?: string;
  compliance_requirements?: string;
  timestamp: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// API CLIENT CLASS
// ─────────────────────────────────────────────────────────────────────────────

export class ApiClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.detail?.message || 
          errorData.detail || 
          `HTTP error! status: ${response.status}`
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // HEALTH & SYSTEM
  // ─────────────────────────────────────────────────────────────────────────────

  async healthCheck(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch {
      return false;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // CHAT API
  // ─────────────────────────────────────────────────────────────────────────────

  async sendMessage(
    message: string, 
    conversationId?: string
  ): Promise<ChatResponse> {
    try {
      // Send message in request body as JSON
      return await this.request<ChatResponse>('/api/chat/', {
        method: 'POST',
        body: JSON.stringify({
          message,
          conversation_id: conversationId
        }),
      });
    } catch {
      // Fallback to multi-agent endpoint
      console.log('Chat endpoint failed, using multi-agent endpoint');
      const response = await this.coordinateTask(message);
      
      // Convert multi-agent response to chat response format
      const chatResponse: ChatResponse = {
        response: response.success 
          ? `Task completed with status: ${(response.result as Record<string, unknown>)?.status || 'unknown'}. ${(response.result as Record<string, unknown>)?.artifacts ? 'Generated artifacts available.' : 'No artifacts generated.'}`
          : `Task failed: ${response.error || 'Unknown error'}`,
        conversation_id: conversationId || response.project_id,
        messages: [
          {
            role: 'user',
            content: message,
            timestamp: new Date().toISOString(),
            metadata: { source: 'frontend' }
          },
          {
            role: 'assistant',
            content: response.success 
              ? `I've processed your request using the multi-agent system. The task "${message}" was ${(response.result as Record<string, unknown>)?.status || 'processed'}. Project ID: ${response.project_id}`
              : `I encountered an error processing your request: ${response.error || 'Unknown error'}`,
            timestamp: response.timestamp,
            metadata: { 
              agent: 'multi-agent-coordinator',
              status: response.success ? 'completed' : 'failed',
              project_id: response.project_id
            }
          }
        ],
        status: response.success ? 'completed' : 'failed',
        active_agent: 'multi-agent-coordinator'
      };
      
      return chatResponse;
    }
  }

  async getChatHistory(conversationId: string): Promise<ChatHistoryResponse> {
    return this.request<ChatHistoryResponse>(`/api/chat/${conversationId}`);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // MULTI-AGENT COORDINATION
  // ─────────────────────────────────────────────────────────────────────────────

  async coordinateTask(
    taskDescription: string, 
    projectId?: string
  ): Promise<MultiAgentResponse> {
    return this.request<MultiAgentResponse>('/multi-agent/coordinate', {
      method: 'POST',
      body: JSON.stringify({
        task_description: taskDescription,
        project_id: projectId,
      }),
    });
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // HIERARCHICAL AGENTS
  // ─────────────────────────────────────────────────────────────────────────────

  async processWithHierarchicalAgents(
    message: string,
    conversationId?: string,
    projectId?: string,
    taskPriority: 'low' | 'medium' | 'high' = 'medium',
    context?: Record<string, unknown>
  ): Promise<{
    response: string;
    conversation_id: string;
    project_id: string;
    agents_used: string[];
    tools_used: string[];
    performance_metrics: Record<string, unknown>;
    human_interventions: number;
  }> {
    return this.request('/hierarchical/process', {
      method: 'POST',
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
        project_id: projectId,
        task_priority: taskPriority,
        context: context || {}
      }),
    });
  }

  async getHierarchicalSystemStatus(): Promise<{
    status: string;
    teams: Array<{
      name: string;
      agents: string[];
      status: string;
    }>;
    infrastructure: Record<string, unknown>;
    performance: Record<string, unknown>;
  }> {
    return this.request('/hierarchical/status');
  }

  async validateHierarchicalInfrastructure(): Promise<{
    overall_status: string;
    component_status: Record<string, unknown>;
    failed_components: string[];
    healthy_components: string[];
  }> {
    return this.request('/hierarchical/validate');
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // AGENTS API
  // ─────────────────────────────────────────────────────────────────────────────

  async getAllAgents(): Promise<Agent[]> {
    return this.request<Agent[]>('/api/agents/');
  }

  async getAgent(agentId: string): Promise<Agent> {
    return this.request<Agent>(`/api/agents/${agentId}`);
  }

  async interactWithAgent(
    agentId: string, 
    message: string
  ): Promise<AgentInteractionResponse> {
    return this.request<AgentInteractionResponse>(
      `/api/agents/${agentId}/interact`, 
      { 
        method: 'POST',
        body: JSON.stringify({ message })
      }
    );
  }

  async getAgentHistory(
    agentId: string, 
    limit: number = 10
  ): Promise<AgentHistoryResponse> {
    const params = new URLSearchParams({ limit: limit.toString() });
    return this.request<AgentHistoryResponse>(
      `/api/agents/${agentId}/history?${params.toString()}`
    );
  }

  async getAgentStatistics(): Promise<AgentStatistics> {
    return this.request<AgentStatistics>('/api/agents/statistics/overview');
  }

  async updateAgentStatus(
    agentId: string, 
    status: Agent['status'], 
    currentTask?: string
  ): Promise<{ success: boolean; message: string }> {
    const body: { status: Agent['status']; current_task?: string } = { status };
    if (currentTask) {
      body.current_task = currentTask;
    }
    
    return this.request(`/api/agents/${agentId}/status`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // DOCUMENTS
  // ─────────────────────────────────────────────────────────────────────────────

  async uploadDocuments(
    files: File[],
    projectId?: string,
    description?: string
  ): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    
    files.forEach((file) => {
      formData.append('files', file);
    });
    
    if (projectId) {
      formData.append('project_id', projectId);
    }
    
    if (description) {
      formData.append('description', description);
    }

    return this.request<DocumentUploadResponse>('/documents/upload', {
      method: 'POST',
      headers: {
        // Don't set Content-Type for FormData, let browser set it with boundary
      },
      body: formData,
    });
  }

  async listDocuments(
    projectId?: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<DocumentListResponse> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
    });
    
    if (projectId) {
      params.append('project_id', projectId);
    }

    return this.request<DocumentListResponse>(`/documents?${params}`);
  }

  async getDocument(documentId: string): Promise<DocumentInfo> {
    return this.request<DocumentInfo>(`/documents/${documentId}`);
  }

  async downloadDocument(documentId: string): Promise<Blob> {
    const url = `${this.baseURL}/documents/${documentId}/download`;
    
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Download failed: ${response.statusText}`);
    }
    
    return response.blob();
  }

  async analyzeDocument(
    documentId: string,
    analysisType: string = 'general'
  ): Promise<DocumentAnalysisResponse> {
    return this.request<DocumentAnalysisResponse>(`/documents/${documentId}/analyze`, {
      method: 'POST',
      body: JSON.stringify({
        analysis_type: analysisType,
      }),
    });
  }

  async convertToTRD(
    documentId: string,
    targetFormat: string = 'technical_requirements',
    stakeholders?: string,
    complianceRequirements?: string
  ): Promise<TRDConversionResponse> {
    const body: Record<string, string> = {
      target_format: targetFormat,
    };
    
    if (stakeholders) {
      body.stakeholders = stakeholders;
    }
    
    if (complianceRequirements) {
      body.compliance_requirements = complianceRequirements;
    }

    return this.request<TRDConversionResponse>(`/documents/${documentId}/convert-trd`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  async deleteDocument(documentId: string): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/documents/${documentId}`, {
      method: 'DELETE',
    });
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // SEARCH & TOOLS
  // ─────────────────────────────────────────────────────────────────────────────

  async searchDocuments(
    query: string, 
    collection: string = 'strategy_book', 
    limit: number = 5
  ) {
    const params = new URLSearchParams({
      query,
      collection,
      limit: limit.toString(),
    });
    
    return this.request(`/api/search/documents?${params.toString()}`);
  }

  async searchWeb(query: string, maxResults: number = 5) {
    const params = new URLSearchParams({
      query,
      max_results: maxResults.toString(),
    });
    
    return this.request(`/api/search/web?${params.toString()}`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLETON INSTANCE
// ─────────────────────────────────────────────────────────────────────────────

export const apiClient = new ApiClient();

// ─────────────────────────────────────────────────────────────────────────────
// CONVENIENCE FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

export const api = {
  // System
  health: () => apiClient.healthCheck(),
  testConnection: () => apiClient.testConnection(),
  
  // Chat
  sendMessage: (message: string, conversationId?: string) => 
    apiClient.sendMessage(message, conversationId),
  getChatHistory: (conversationId: string) => 
    apiClient.getChatHistory(conversationId),
  
  // Multi-agent
  coordinateTask: (taskDescription: string, projectId?: string) => 
    apiClient.coordinateTask(taskDescription, projectId),
  
  // Agents
  agents: {
    getAll: () => apiClient.getAllAgents(),
    get: (id: string) => apiClient.getAgent(id),
    interact: (id: string, message: string) => apiClient.interactWithAgent(id, message),
    getHistory: (id: string, limit?: number) => apiClient.getAgentHistory(id, limit),
    getStatistics: () => apiClient.getAgentStatistics(),
    updateStatus: (id: string, status: Agent['status'], currentTask?: string) => 
      apiClient.updateAgentStatus(id, status, currentTask),
  },
  
  // Documents
  documents: {
    upload: (files: File[], projectId?: string, description?: string) => 
      apiClient.uploadDocuments(files, projectId, description),
    list: (projectId?: string, limit?: number, offset?: number) => 
      apiClient.listDocuments(projectId, limit, offset),
    get: (documentId: string) => apiClient.getDocument(documentId),
    download: (documentId: string) => apiClient.downloadDocument(documentId),
    analyze: (documentId: string, analysisType?: string) => 
      apiClient.analyzeDocument(documentId, analysisType),
    convertToTRD: (documentId: string, targetFormat?: string, stakeholders?: string, complianceRequirements?: string) => 
      apiClient.convertToTRD(documentId, targetFormat, stakeholders, complianceRequirements),
    delete: (documentId: string) => apiClient.deleteDocument(documentId),
  },
  
  // Search
  search: {
    documents: (query: string, collection?: string, limit?: number) => 
      apiClient.searchDocuments(query, collection, limit),
    web: (query: string, maxResults?: number) => 
      apiClient.searchWeb(query, maxResults),
  },
};

export default api;

// React hooks for API integration
export { useChat } from '../hooks/useChat'; 