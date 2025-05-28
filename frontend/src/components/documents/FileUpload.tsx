/**
 * FileUpload Component
 * Handles file upload with drag-and-drop functionality, progress tracking,
 * and integration with the Documentation team for TRD processing
 */

'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useCallback, useState } from 'react';
import { 
  CloudArrowUpIcon, 
  DocumentIcon, 
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  CogIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';
import useDocuments from '../../hooks/useDocuments';
import { DocumentUploadResponse } from '../../utils/api';

interface FileUploadProps {
  onUploadComplete?: (response: DocumentUploadResponse) => void;
  onClose?: () => void;
  projectId?: string;
  maxFiles?: number;
  maxFileSize?: number; // in MB
  acceptedTypes?: string[];
  showTRDOptions?: boolean;
}

interface UploadFile extends File {
  id: string;
  progress: number;
  status: 'pending' | 'uploading' | 'completed' | 'error';
  error?: string;
}

export default function FileUpload({
  onUploadComplete,
  onClose,
  projectId,
  maxFiles = 10,
  maxFileSize = 50,
  acceptedTypes = ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf'],
  showTRDOptions = true
}: FileUploadProps) {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [description, setDescription] = useState('');
  const [enableTRDConversion, setEnableTRDConversion] = useState(false);
  const [stakeholders, setStakeholders] = useState('');
  const [complianceRequirements, setComplianceRequirements] = useState('');

  const {
    uploadDocuments,
    isUploading,
    error,
    lastUploadResponse,
    lastTRDResponse,
    convertToTRD,
    clearError
  } = useDocuments(projectId);

  const validateFile = useCallback((file: File): string | null => {
    // Check file size
    if (file.size > maxFileSize * 1024 * 1024) {
      return `File size exceeds ${maxFileSize}MB limit`;
    }

    // Check file type
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedTypes.includes(fileExtension)) {
      return `File type not supported. Accepted types: ${acceptedTypes.join(', ')}`;
    }

    return null;
  }, [maxFileSize, acceptedTypes]);

  const addFiles = useCallback((newFiles: FileList | File[]) => {
    const fileArray = Array.from(newFiles);
    
    if (files.length + fileArray.length > maxFiles) {
      alert(`Maximum ${maxFiles} files allowed`);
      return;
    }

    const validFiles: UploadFile[] = [];
    
    fileArray.forEach(file => {
      const error = validateFile(file);
      const uploadFile: UploadFile = {
        ...file,
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        progress: 0,
        status: error ? 'error' : 'pending',
        error: error || undefined
      };
      validFiles.push(uploadFile);
    });

    setFiles(prev => [...prev, ...validFiles]);
  }, [files.length, maxFiles, validateFile]);

  const removeFile = useCallback((fileId: string) => {
    setFiles(prev => prev.filter(f => f.id !== fileId));
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles.length > 0) {
      addFiles(droppedFiles);
    }
  }, [addFiles]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (selectedFiles && selectedFiles.length > 0) {
      addFiles(selectedFiles);
    }
    // Reset input value to allow selecting the same file again
    e.target.value = '';
  }, [addFiles]);

  const handleUpload = useCallback(async () => {
    const validFiles = files.filter(f => f.status !== 'error');
    if (validFiles.length === 0) return;

    clearError();

    try {
      // Update file status to uploading
      setFiles(prev => prev.map(f => 
        f.status !== 'error' ? { ...f, status: 'uploading' as const, progress: 0 } : f
      ));

      // Convert UploadFile[] back to File[] for API
      const fileObjects = validFiles.map(f => {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const { id, progress, status, error: fileError, ...fileProps } = f;
        return new File([f], f.name, { type: f.type, lastModified: f.lastModified });
      });

      const response = await uploadDocuments(fileObjects, projectId, description);

      if (response.success) {
        // Update file status to completed
        setFiles(prev => prev.map(f => 
          f.status === 'uploading' ? { ...f, status: 'completed' as const, progress: 100 } : f
        ));

        // If TRD conversion is enabled, convert each uploaded document
        if (enableTRDConversion && response.uploaded_files.length > 0) {
          for (const uploadedFile of response.uploaded_files) {
            try {
              await convertToTRD(
                uploadedFile.id,
                'technical_requirements',
                stakeholders || undefined,
                complianceRequirements || undefined
              );
            } catch (trdError) {
              console.error(`TRD conversion failed for ${uploadedFile.filename}:`, trdError);
            }
          }
        }

        onUploadComplete?.(response);
      }
    } catch (uploadError) {
      // Update file status to error
      setFiles(prev => prev.map(f => 
        f.status === 'uploading' ? { 
          ...f, 
          status: 'error' as const, 
          progress: 0,
          error: uploadError instanceof Error ? uploadError.message : 'Upload failed'
        } : f
      ));
    }
  }, [files, projectId, description, enableTRDConversion, stakeholders, complianceRequirements, uploadDocuments, convertToTRD, clearError, onUploadComplete]);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: UploadFile['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'error':
        return <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />;
      case 'uploading':
        return <CogIcon className="w-5 h-5 text-blue-500 animate-spin" />;
      default:
        return <DocumentIcon className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: UploadFile['status']) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'uploading':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const canUpload = files.some(f => f.status === 'pending') && !isUploading;
  const hasValidFiles = files.some(f => f.status !== 'error');

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <CloudArrowUpIcon className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Upload Documents</h2>
              <p className="text-sm text-gray-500">
                Upload files for processing and analysis
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <XMarkIcon className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {/* Upload Area */}
          <div
            className={`
              relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200
              ${isDragOver 
                ? 'border-blue-400 bg-blue-50' 
                : 'border-gray-300 hover:border-gray-400'
              }
            `}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              type="file"
              multiple
              accept={acceptedTypes.join(',')}
              onChange={handleFileSelect}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            
            <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Drop files here or click to browse
            </h3>
            <p className="text-sm text-gray-500 mb-4">
              Supports: {acceptedTypes.join(', ')} • Max {maxFileSize}MB per file • Up to {maxFiles} files
            </p>
            
            <button
              type="button"
              className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <DocumentIcon className="w-4 h-4 mr-2" />
              Choose Files
            </button>
          </div>

          {/* Description */}
          <div className="mt-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Description (Optional)
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe the purpose or content of these documents..."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
              rows={3}
            />
          </div>

          {/* TRD Conversion Options */}
          {showTRDOptions && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-3 mb-4">
                <input
                  type="checkbox"
                  id="enableTRD"
                  checked={enableTRDConversion}
                  onChange={(e) => setEnableTRDConversion(e.target.checked)}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="enableTRD" className="flex items-center space-x-2">
                  <DocumentTextIcon className="w-5 h-5 text-gray-600" />
                  <span className="text-sm font-medium text-gray-700">
                    Convert to Technical Requirements Document (TRD)
                  </span>
                </label>
              </div>

              <AnimatePresence>
                {enableTRDConversion && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-4"
                  >
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Stakeholders
                      </label>
                      <input
                        type="text"
                        value={stakeholders}
                        onChange={(e) => setStakeholders(e.target.value)}
                        placeholder="e.g., Development Team, Product Manager, QA Team"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Compliance Requirements
                      </label>
                      <input
                        type="text"
                        value={complianceRequirements}
                        onChange={(e) => setComplianceRequirements(e.target.value)}
                        placeholder="e.g., GDPR, HIPAA, SOX, ISO 27001"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* File List */}
          {files.length > 0 && (
            <div className="mt-6">
              <h4 className="text-sm font-medium text-gray-700 mb-3">
                Selected Files ({files.length})
              </h4>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                <AnimatePresence>
                  {files.map((file) => (
                    <motion.div
                      key={file.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className={`
                        flex items-center justify-between p-3 rounded-lg border
                        ${getStatusColor(file.status)}
                      `}
                    >
                      <div className="flex items-center space-x-3 flex-1 min-w-0">
                        {getStatusIcon(file.status)}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">
                            {file.name}
                          </p>
                          <p className="text-xs text-gray-500">
                            {formatFileSize(file.size)}
                          </p>
                          {file.error && (
                            <p className="text-xs text-red-600 mt-1">
                              {file.error}
                            </p>
                          )}
                        </div>
                      </div>
                      
                      {file.status === 'uploading' && (
                        <div className="w-20">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${file.progress}%` }}
                            />
                          </div>
                        </div>
                      )}
                      
                      {file.status !== 'uploading' && (
                        <button
                          onClick={() => removeFile(file.id)}
                          className="p-1 hover:bg-gray-200 rounded transition-colors"
                        >
                          <XMarkIcon className="w-4 h-4 text-gray-500" />
                        </button>
                      )}
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center space-x-2">
                <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          )}

          {/* Success Messages */}
          {lastUploadResponse?.success && (
            <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center space-x-2">
                <CheckCircleIcon className="w-5 h-5 text-green-500" />
                <p className="text-sm text-green-700">
                  Successfully uploaded {lastUploadResponse.total_uploaded} file(s)
                </p>
              </div>
            </div>
          )}

          {lastTRDResponse?.success && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center space-x-2">
                <DocumentTextIcon className="w-5 h-5 text-blue-500" />
                <p className="text-sm text-blue-700">
                  TRD conversion completed successfully
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-gray-200 bg-gray-50">
          <div className="text-sm text-gray-500">
            {files.length > 0 && (
              <span>
                {files.filter(f => f.status === 'completed').length} completed, {' '}
                {files.filter(f => f.status === 'error').length} failed, {' '}
                {files.filter(f => f.status === 'pending').length} pending
              </span>
            )}
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleUpload}
              disabled={!canUpload || !hasValidFiles}
              className={`
                px-6 py-2 rounded-lg font-medium transition-colors
                ${canUpload && hasValidFiles
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }
              `}
            >
              {isUploading ? (
                <div className="flex items-center space-x-2">
                  <CogIcon className="w-4 h-4 animate-spin" />
                  <span>Uploading...</span>
                </div>
              ) : (
                'Upload Files'
              )}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
} 