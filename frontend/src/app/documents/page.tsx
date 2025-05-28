'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Plus, Filter, Search, Grid, List } from 'lucide-react';
import { Layout } from '@/components/layout/Layout';
import { DocumentGrid, DocumentList } from '@/components/documents/DocumentCard';
import FileUpload from '@/components/documents/FileUpload';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { GlassCard } from '@/components/ui/GlassCard';
import { Modal } from '@/components/ui/Modal';
import { useToastActions } from '@/components/ui/Toast';
import { DocumentUploadResponse } from '@/utils/api';
import clsx from 'clsx';

interface Document {
  id: string;
  name: string;
  type: 'text' | 'image' | 'video' | 'archive' | 'other';
  size: number;
  createdAt: Date;
  updatedAt: Date;
  author: {
    id: string;
    name: string;
    avatar?: string;
  };
  tags: string[];
  description?: string;
  thumbnail?: string;
  url: string;
  isShared: boolean;
  downloadCount: number;
  version: number;
}

export default function DocumentsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const toast = useToastActions();

  // Fetch documents data
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        setLoading(true);
        
        // TODO: Replace with actual API call
        // const response = await fetch('/api/documents');
        // const data = await response.json();
        // setDocuments(data);
        
        // For now, set empty data
        setDocuments([]);
        
      } catch (error) {
        console.error('Failed to fetch documents:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDocuments();
  }, []);

  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         doc.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         doc.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesType = typeFilter === 'all' || doc.type === typeFilter;
    return matchesSearch && matchesType;
  });

  const handleDocumentView = (documentId: string) => {
    console.log('Viewing document:', documentId);
    // TODO: Implement document viewer
  };

  const handleDocumentEdit = (documentId: string) => {
    console.log('Editing document:', documentId);
    // TODO: Implement document editor
  };

  const handleDocumentDownload = (documentId: string) => {
    console.log('Downloading document:', documentId);
    // TODO: Implement document download
  };

  const handleDocumentShare = (documentId: string) => {
    console.log('Sharing document:', documentId);
    // TODO: Implement document sharing
  };

  const handleFileUploadComplete = (response: DocumentUploadResponse) => {
    console.log('Upload completed:', response);
    toast.success('Files uploaded', `Upload completed successfully`);
    setShowUploadModal(false);
    
    // TODO: Refresh the documents list from the backend
    // For now, we'll just close the modal
  };

  const typeCounts = {
    all: documents.length,
    text: documents.filter(d => d.type === 'text').length,
    image: documents.filter(d => d.type === 'image').length,
    video: documents.filter(d => d.type === 'video').length,
    archive: documents.filter(d => d.type === 'archive').length,
    other: documents.filter(d => d.type === 'other').length
  };

  if (loading) {
    return (
      <Layout 
        sidebarCollapsed={sidebarCollapsed}
        onSidebarToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      >
        <div className="flex items-center justify-center h-64">
          <div className="loading loading-spinner loading-lg"></div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout 
      sidebarCollapsed={sidebarCollapsed}
      onSidebarToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
    >
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-4xl font-bold gradient-text mb-2">
              Documents
            </h1>
            <p className="text-gray-600 text-lg">
              Manage your files, reports, and generated content
            </p>
          </div>
          <Button 
            leftIcon={<Plus className="w-5 h-5" />}
            onClick={() => setShowUploadModal(true)}
          >
            Upload Document
          </Button>
        </div>

        {/* Stats and Filters */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
          {Object.entries(typeCounts).map(([type, count]) => (
            <GlassCard
              key={type}
              hover={true}
              padding="sm"
              className={clsx(
                'cursor-pointer transition-all duration-200',
                typeFilter === type && 'ring-2 ring-blue-500'
              )}
              onClick={() => setTypeFilter(type)}
            >
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900">{count}</p>
                <p className="text-sm text-gray-600 capitalize">{type}</p>
              </div>
            </GlassCard>
          ))}
        </div>

        {/* Search and View Controls */}
        <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
          <div className="flex-1 max-w-md">
            <Input
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={<Search className="w-5 h-5" />}
            />
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" leftIcon={<Filter className="w-5 h-5" />}>
              Filters
            </Button>
            <div className="flex items-center bg-white/20 rounded-lg p-1">
              <Button
                variant={viewMode === 'grid' ? 'primary' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('grid')}
                className="!p-2"
              >
                <Grid className="w-4 h-4" />
              </Button>
              <Button
                variant={viewMode === 'list' ? 'primary' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('list')}
                className="!p-2"
              >
                <List className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Documents Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {filteredDocuments.length === 0 ? (
          <GlassCard>
            <div className="text-center py-12">
              <div className="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <Plus className="w-12 h-12 text-gray-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                No documents yet
              </h3>
              <p className="text-gray-600 mb-6 max-w-md mx-auto">
                Upload your first document to get started. You can upload files, images, videos, and more.
              </p>
              <Button 
                leftIcon={<Plus className="w-5 h-5" />}
                onClick={() => setShowUploadModal(true)}
              >
                Upload Your First Document
              </Button>
            </div>
          </GlassCard>
        ) : (
          <>
            {viewMode === 'grid' ? (
              <DocumentGrid
                documents={filteredDocuments}
                onView={handleDocumentView}
                onEdit={handleDocumentEdit}
                onDownload={handleDocumentDownload}
                onShare={handleDocumentShare}
              />
            ) : (
              <DocumentList
                documents={filteredDocuments}
                onView={handleDocumentView}
                onEdit={handleDocumentEdit}
                onDownload={handleDocumentDownload}
                onShare={handleDocumentShare}
              />
            )}
          </>
        )}
      </motion.div>

      {/* Upload Modal */}
      <Modal
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        title="Upload Documents"
        size="lg"
      >
        <FileUpload
          onUploadComplete={handleFileUploadComplete}
          onClose={() => setShowUploadModal(false)}
          maxFiles={10}
          maxFileSize={50}
          acceptedTypes={['*/*']}
        />
      </Modal>
    </Layout>
  );
} 