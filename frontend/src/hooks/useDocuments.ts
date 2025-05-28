/**
 * useDocuments Hook
 * Manages document upload, listing, analysis, and TRD conversion operations
 */

import { useCallback, useState } from 'react';
import { 
  api, 
  DocumentUploadResponse, 
  DocumentListResponse, 
  DocumentInfo, 
  DocumentAnalysisResponse, 
  TRDConversionResponse 
} from '../utils/api';

interface UseDocumentsState {
  documents: DocumentInfo[];
  isLoading: boolean;
  isUploading: boolean;
  isAnalyzing: boolean;
  isConverting: boolean;
  error: string | null;
  uploadProgress: number;
  lastUploadResponse: DocumentUploadResponse | null;
  lastAnalysisResponse: DocumentAnalysisResponse | null;
  lastTRDResponse: TRDConversionResponse | null;
}

interface UseDocumentsReturn extends UseDocumentsState {
  uploadDocuments: (files: File[], projectId?: string, description?: string) => Promise<DocumentUploadResponse>;
  listDocuments: (projectId?: string, limit?: number, offset?: number) => Promise<DocumentListResponse>;
  refreshDocuments: () => Promise<void>;
  analyzeDocument: (documentId: string, analysisType?: string) => Promise<DocumentAnalysisResponse>;
  convertToTRD: (
    documentId: string, 
    targetFormat?: string, 
    stakeholders?: string, 
    complianceRequirements?: string
  ) => Promise<TRDConversionResponse>;
  downloadDocument: (documentId: string, filename?: string) => Promise<void>;
  deleteDocument: (documentId: string) => Promise<void>;
  clearError: () => void;
  clearResponses: () => void;
}

export const useDocuments = (initialProjectId?: string): UseDocumentsReturn => {
  const [state, setState] = useState<UseDocumentsState>({
    documents: [],
    isLoading: false,
    isUploading: false,
    isAnalyzing: false,
    isConverting: false,
    error: null,
    uploadProgress: 0,
    lastUploadResponse: null,
    lastAnalysisResponse: null,
    lastTRDResponse: null,
  });

  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  const clearResponses = useCallback(() => {
    setState(prev => ({
      ...prev,
      lastUploadResponse: null,
      lastAnalysisResponse: null,
      lastTRDResponse: null,
    }));
  }, []);

  const listDocuments = useCallback(async (
    projectId?: string, 
    limit: number = 50, 
    offset: number = 0
  ): Promise<DocumentListResponse> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const response = await api.documents.list(projectId || initialProjectId, limit, offset);
      
      setState(prev => ({
        ...prev,
        isLoading: false,
        documents: response.documents,
      }));

      return response;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load documents',
      }));
      throw error;
    }
  }, [initialProjectId]);

  const uploadDocuments = useCallback(async (
    files: File[], 
    projectId?: string, 
    description?: string
  ): Promise<DocumentUploadResponse> => {
    setState(prev => ({ 
      ...prev, 
      isUploading: true, 
      error: null, 
      uploadProgress: 0 
    }));

    try {
      // Simulate upload progress (since we can't track real progress with current API)
      const progressInterval = setInterval(() => {
        setState(prev => ({
          ...prev,
          uploadProgress: Math.min(prev.uploadProgress + 10, 90)
        }));
      }, 200);

      const response = await api.documents.upload(files, projectId || initialProjectId, description);
      
      clearInterval(progressInterval);
      
      setState(prev => ({
        ...prev,
        isUploading: false,
        uploadProgress: 100,
        lastUploadResponse: response,
      }));

      // Auto-refresh documents list after successful upload
      if (response.success && response.total_uploaded > 0) {
        setTimeout(async () => {
          try {
            await listDocuments();
          } catch (error) {
            console.error('Failed to refresh documents:', error);
          }
        }, 500);
      }

      return response;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isUploading: false,
        uploadProgress: 0,
        error: error instanceof Error ? error.message : 'Upload failed',
      }));
      throw error;
    }
  }, [initialProjectId, listDocuments]);

  const refreshDocuments = useCallback(async (): Promise<void> => {
    try {
      await listDocuments();
    } catch (error) {
      console.error('Failed to refresh documents:', error);
    }
  }, [listDocuments]);

  const analyzeDocument = useCallback(async (
    documentId: string, 
    analysisType: string = 'general'
  ): Promise<DocumentAnalysisResponse> => {
    setState(prev => ({ ...prev, isAnalyzing: true, error: null }));

    try {
      const response = await api.documents.analyze(documentId, analysisType);
      
      setState(prev => ({
        ...prev,
        isAnalyzing: false,
        lastAnalysisResponse: response,
      }));

      return response;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isAnalyzing: false,
        error: error instanceof Error ? error.message : 'Analysis failed',
      }));
      throw error;
    }
  }, []);

  const convertToTRD = useCallback(async (
    documentId: string,
    targetFormat: string = 'technical_requirements',
    stakeholders?: string,
    complianceRequirements?: string
  ): Promise<TRDConversionResponse> => {
    setState(prev => ({ ...prev, isConverting: true, error: null }));

    try {
      const response = await api.documents.convertToTRD(
        documentId, 
        targetFormat, 
        stakeholders, 
        complianceRequirements
      );
      
      setState(prev => ({
        ...prev,
        isConverting: false,
        lastTRDResponse: response,
      }));

      return response;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isConverting: false,
        error: error instanceof Error ? error.message : 'TRD conversion failed',
      }));
      throw error;
    }
  }, []);

  const downloadDocument = useCallback(async (
    documentId: string, 
    filename?: string
  ): Promise<void> => {
    try {
      const blob = await api.documents.download(documentId);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename || `document-${documentId}`;
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Download failed',
      }));
      throw error;
    }
  }, []);

  const deleteDocument = useCallback(async (documentId: string): Promise<void> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      await api.documents.delete(documentId);
      
      // Remove document from local state
      setState(prev => ({
        ...prev,
        isLoading: false,
        documents: prev.documents.filter(doc => doc.id !== documentId),
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Delete failed',
      }));
      throw error;
    }
  }, []);

  return {
    ...state,
    uploadDocuments,
    listDocuments,
    refreshDocuments,
    analyzeDocument,
    convertToTRD,
    downloadDocument,
    deleteDocument,
    clearError,
    clearResponses,
  };
};

export default useDocuments; 