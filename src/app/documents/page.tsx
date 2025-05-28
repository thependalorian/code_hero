'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  PlusIcon, 
  FunnelIcon, 
  MagnifyingGlassIcon, 
  Squares2X2Icon, 
  ListBulletIcon,
  DocumentIcon,
  CloudArrowDownIcon,
  TrashIcon,
  CogIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';
import Layout from '@/components/layout/Layout';
import FileUpload from '@/components/documents/FileUpload';
import Modal from '@/components/ui/Modal';
import useDocuments from '@/hooks/useDocuments';
import { DocumentInfo } from '@/utils/api';

export default function DocumentsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<DocumentInfo | null>(null);

  const {
    documents,
    isLoading,
    error,
    listDocuments,
    refreshDocuments,
    analyzeDocument,
    convertToTRD,
    downloadDocument,
    deleteDocument,
    isAnalyzing,
    isConverting,
    lastAnalysisResponse,
    lastTRDResponse,
    clearError
  } = useDocuments();

  // Fetch documents on component mount
  useEffect(() => {
    listDocuments();
  }, [listDocuments]);

  // Filter documents based on search and type
  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         doc.description?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = typeFilter === 'all' || doc.format === typeFilter;
    return matchesSearch && matchesType;
  });

  // Get unique file types for filter
  const fileTypes = ['all', ...new Set(documents.map(doc => doc.format))];

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getStatusBadge = (status: string) => {
    const statusColors = {
      'uploaded': 'badge-success',
      'processing': 'badge-warning',
      'analyzed': 'badge-info',
      'converted': 'badge-primary',
      'error': 'badge-error'
    };
    
    return (
      <span className={`badge badge-sm ${statusColors[status as keyof typeof statusColors] || 'badge-ghost'}`}>
        {status}
      </span>
    );
  };

  const handleUploadComplete = (response: any) => {
    setShowUploadModal(false);
    refreshDocuments();
  };

  const handleAnalyze = async (document: DocumentInfo) => {
    try {
      await analyzeDocument(document.id, 'general');
    } catch (error) {
      console.error('Analysis failed:', error);
    }
  };

  const handleConvertToTRD = async (document: DocumentInfo) => {
    try {
      await convertToTRD(document.id, 'technical_requirements');
    } catch (error) {
      console.error('TRD conversion failed:', error);
    }
  };

  const handleDownload = async (document: DocumentInfo) => {
    try {
      await downloadDocument(document.id, document.filename);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const handleDelete = async (document: DocumentInfo) => {
    if (confirm(`Are you sure you want to delete "${document.filename}"?`)) {
      try {
        await deleteDocument(document.id);
      } catch (error) {
        console.error('Delete failed:', error);
      }
    }
  };

  return (
    <Layout>
      <div className="min-h-screen bg-base-100">
        {/* Header */}
        <div className="bg-base-200 border-b border-base-300">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-base-content">Documents</h1>
                <p className="text-base-content/60 mt-1">
                  Manage and process documents with AI-powered analysis and TRD conversion
                </p>
              </div>
              <button
                onClick={() => setShowUploadModal(true)}
                className="btn btn-primary gap-2"
              >
                <PlusIcon className="w-5 h-5" />
                Upload Documents
              </button>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div className="alert alert-error">
              <span>{error}</span>
              <button onClick={clearError} className="btn btn-ghost btn-sm">
                Dismiss
              </button>
            </div>
          </div>
        )}

        {/* Analysis/TRD Results */}
        {(lastAnalysisResponse || lastTRDResponse) && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            {lastAnalysisResponse && (
              <div className="alert alert-info mb-4">
                <DocumentIcon className="w-5 h-5" />
                <div>
                  <div className="font-medium">Document Analysis Complete</div>
                  <div className="text-sm opacity-80">
                    Analysis result available for document {lastAnalysisResponse.document_id}
                  </div>
                </div>
              </div>
            )}
            {lastTRDResponse && (
              <div className="alert alert-success">
                <DocumentTextIcon className="w-5 h-5" />
                <div>
                  <div className="font-medium">TRD Conversion Complete</div>
                  <div className="text-sm opacity-80">
                    Technical Requirements Document generated for {lastTRDResponse.document_id}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Controls */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
            {/* Search and Filter */}
            <div className="flex flex-1 gap-4 max-w-2xl">
              <div className="relative flex-1">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-base-content/40" />
                <input
                  type="text"
                  placeholder="Search documents..."
                  className="input input-bordered w-full pl-10"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              <select
                className="select select-bordered"
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
              >
                {fileTypes.map(type => (
                  <option key={type} value={type}>
                    {type === 'all' ? 'All Types' : type.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            {/* View Mode Toggle */}
            <div className="join">
              <button
                className={`btn join-item ${viewMode === 'grid' ? 'btn-active' : 'btn-ghost'}`}
                onClick={() => setViewMode('grid')}
              >
                <Squares2X2Icon className="w-5 h-5" />
              </button>
              <button
                className={`btn join-item ${viewMode === 'list' ? 'btn-active' : 'btn-ghost'}`}
                onClick={() => setViewMode('list')}
              >
                <ListBulletIcon className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Documents Display */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-8">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="loading loading-spinner loading-lg"></div>
              <span className="ml-3 text-base-content/60">Loading documents...</span>
            </div>
          ) : filteredDocuments.length === 0 ? (
            <div className="text-center py-12">
              <DocumentIcon className="w-16 h-16 mx-auto text-base-content/40 mb-4" />
              <h3 className="text-lg font-medium text-base-content mb-2">
                {documents.length === 0 ? 'No documents uploaded' : 'No documents match your search'}
              </h3>
              <p className="text-base-content/60 mb-6">
                {documents.length === 0 
                  ? 'Upload your first document to get started with AI-powered analysis and TRD conversion.'
                  : 'Try adjusting your search terms or filters.'
                }
              </p>
              {documents.length === 0 && (
                <button
                  onClick={() => setShowUploadModal(true)}
                  className="btn btn-primary"
                >
                  <PlusIcon className="w-5 h-5 mr-2" />
                  Upload Documents
                </button>
              )}
            </div>
          ) : (
            <div className={viewMode === 'grid' 
              ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6'
              : 'space-y-4'
            }>
              {filteredDocuments.map((document) => (
                <motion.div
                  key={document.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={viewMode === 'grid' 
                    ? 'card bg-base-200 shadow-lg hover:shadow-xl transition-shadow'
                    : 'card bg-base-200 shadow-sm hover:shadow-md transition-shadow'
                  }
                >
                  <div className="card-body">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium text-base-content truncate">
                          {document.filename}
                        </h3>
                        <p className="text-sm text-base-content/60 mt-1">
                          {formatFileSize(document.size)} • {document.format.toUpperCase()}
                        </p>
                        <p className="text-xs text-base-content/50 mt-1">
                          {formatDate(document.upload_timestamp)}
                        </p>
                        {document.description && (
                          <p className="text-sm text-base-content/70 mt-2 line-clamp-2">
                            {document.description}
                          </p>
                        )}
                      </div>
                      <div className="ml-3">
                        {getStatusBadge(document.processing_status)}
                      </div>
                    </div>

                    <div className="card-actions justify-end mt-4">
                      <div className="dropdown dropdown-end">
                        <label tabIndex={0} className="btn btn-ghost btn-sm">
                          Actions
                        </label>
                        <ul tabIndex={0} className="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-52">
                          <li>
                            <button onClick={() => handleDownload(document)}>
                              <CloudArrowDownIcon className="w-4 h-4" />
                              Download
                            </button>
                          </li>
                          <li>
                            <button 
                              onClick={() => handleAnalyze(document)}
                              disabled={isAnalyzing}
                            >
                              <CogIcon className={`w-4 h-4 ${isAnalyzing ? 'animate-spin' : ''}`} />
                              Analyze
                            </button>
                          </li>
                          <li>
                            <button 
                              onClick={() => handleConvertToTRD(document)}
                              disabled={isConverting}
                            >
                              <DocumentTextIcon className={`w-4 h-4 ${isConverting ? 'animate-spin' : ''}`} />
                              Convert to TRD
                            </button>
                          </li>
                          <li>
                            <button 
                              onClick={() => handleDelete(document)}
                              className="text-error"
                            >
                              <TrashIcon className="w-4 h-4" />
                              Delete
                            </button>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>

        {/* Upload Modal */}
        <Modal
          isOpen={showUploadModal}
          onClose={() => setShowUploadModal(false)}
          size="large"
        >
          <FileUpload
            onUploadComplete={handleUploadComplete}
            onClose={() => setShowUploadModal(false)}
            showTRDOptions={true}
          />
        </Modal>
      </div>
    </Layout>
  );
} 