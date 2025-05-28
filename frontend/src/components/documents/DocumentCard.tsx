'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  FileText, 
  File, 
  Image as ImageIcon, 
  Video, 
  Music, 
  Archive,
  Download,
  Share,
  Edit,
  Eye,
  Calendar,
  User,
  Tag
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import { Button } from '../ui/Button';
import { clsx } from 'clsx';
import Image from 'next/image';

interface Document {
  id: string;
  name: string;
  type: 'text' | 'image' | 'video' | 'audio' | 'archive' | 'other';
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

interface DocumentCardProps {
  document: Document;
  onView?: (documentId: string) => void;
  onEdit?: (documentId: string) => void;
  onDownload?: (documentId: string) => void;
  onShare?: (documentId: string) => void;
  onDelete?: (documentId: string) => void;
  className?: string;
}

const fileTypeIcons = {
  text: FileText,
  image: ImageIcon,
  video: Video,
  audio: Music,
  archive: Archive,
  other: File
};

const fileTypeColors = {
  text: 'from-blue-500 to-blue-600',
  image: 'from-green-500 to-green-600',
  video: 'from-red-500 to-red-600',
  audio: 'from-purple-500 to-purple-600',
  archive: 'from-orange-500 to-orange-600',
  other: 'from-gray-500 to-gray-600'
};

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const DocumentCard: React.FC<DocumentCardProps> = ({
  document,
  onView,
  onEdit,
  onDownload,
  onShare,
  onDelete: _onDelete, // eslint-disable-line @typescript-eslint/no-unused-vars
  className = ''
}) => {
  const IconComponent = fileTypeIcons[document.type];
  const gradient = fileTypeColors[document.type];

  return (
    <GlassCard hover={true} className={clsx('h-full', className)}>
      <div className="space-y-4">
        {/* Thumbnail/Icon */}
        <div className="relative">
          {document.thumbnail ? (
            <div className="w-full h-32 rounded-lg overflow-hidden bg-gray-100">
              <Image
                src={document.thumbnail}
                alt={document.name}
                width={300}
                height={128}
                className="w-full h-full object-cover"
              />
            </div>
          ) : (
            <div className={`w-full h-32 bg-gradient-to-br ${gradient} rounded-lg flex items-center justify-center`}>
              <IconComponent className="w-12 h-12 text-white" />
            </div>
          )}
          
          {/* File Type Badge */}
          <div className="absolute top-2 right-2">
            <span className="text-xs font-medium px-2 py-1 bg-black/50 text-white rounded-full backdrop-blur-sm">
              {document.type.toUpperCase()}
            </span>
          </div>

          {/* Shared Badge */}
          {document.isShared && (
            <div className="absolute top-2 left-2">
              <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
                <Share className="w-3 h-3 text-white" />
              </div>
            </div>
          )}
        </div>

        {/* Document Info */}
        <div className="space-y-2">
          <h3 className="font-semibold text-gray-900 line-clamp-2" title={document.name}>
            {document.name}
          </h3>
          
          {document.description && (
            <p className="text-sm text-gray-600 line-clamp-2">
              {document.description}
            </p>
          )}

          {/* Metadata */}
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>{formatFileSize(document.size)}</span>
              <span>v{document.version}</span>
            </div>
            
            <div className="flex items-center space-x-2 text-xs text-gray-500">
              <Calendar className="w-3 h-3" />
              <span>{document.updatedAt.toLocaleDateString()}</span>
            </div>
            
            <div className="flex items-center space-x-2 text-xs text-gray-500">
              <User className="w-3 h-3" />
              <span>{document.author.name}</span>
            </div>

            <div className="flex items-center space-x-2 text-xs text-gray-500">
              <Download className="w-3 h-3" />
              <span>{document.downloadCount} downloads</span>
            </div>
          </div>
        </div>

        {/* Tags */}
        {document.tags.length > 0 && (
          <div>
            <div className="flex items-center space-x-1 mb-2">
              <Tag className="w-3 h-3 text-gray-400" />
              <span className="text-xs font-medium text-gray-500">Tags</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {document.tags.slice(0, 3).map((tag, index) => (
                <span
                  key={index}
                  className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full"
                >
                  {tag}
                </span>
              ))}
              {document.tags.length > 3 && (
                <span className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded-full">
                  +{document.tags.length - 3}
                </span>
              )}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex space-x-1 pt-2">
          <Button
            variant="primary"
            size="sm"
            className="flex-1"
            onClick={() => onView?.(document.id)}
            leftIcon={<Eye className="w-3 h-3" />}
          >
            View
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onDownload?.(document.id)}
          >
            <Download className="w-3 h-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onShare?.(document.id)}
          >
            <Share className="w-3 h-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onEdit?.(document.id)}
          >
            <Edit className="w-3 h-3" />
          </Button>
        </div>
      </div>
    </GlassCard>
  );
};

export const DocumentGrid: React.FC<{
  documents: Document[];
  onView?: (documentId: string) => void;
  onEdit?: (documentId: string) => void;
  onDownload?: (documentId: string) => void;
  onShare?: (documentId: string) => void;
  onDelete?: (documentId: string) => void;
  className?: string;
}> = ({ documents, onView, onEdit, onDownload, onShare, onDelete: _onDelete, className = '' }) => {
  return (
    <div className={clsx('grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6', className)}>
      {documents.map((document, index) => (
        <motion.div
          key={document.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <DocumentCard
            document={document}
            onView={onView}
            onEdit={onEdit}
            onDownload={onDownload}
            onShare={onShare}
            onDelete={_onDelete}
          />
        </motion.div>
      ))}
    </div>
  );
};

export const DocumentList: React.FC<{
  documents: Document[];
  onView?: (documentId: string) => void;
  onEdit?: (documentId: string) => void;
  onDownload?: (documentId: string) => void;
  onShare?: (documentId: string) => void;
  onDelete?: (documentId: string) => void;
  className?: string;
}> = ({ documents, onView, onEdit: _onEdit, onDownload, onShare, onDelete: _onDelete, className = '' }) => { // eslint-disable-line @typescript-eslint/no-unused-vars
  return (
    <div className={clsx('space-y-3', className)}>
      {documents.map((document, index) => {
        const IconComponent = fileTypeIcons[document.type];
        const gradient = fileTypeColors[document.type];
        
        return (
          <motion.div
            key={document.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <GlassCard hover={true} padding="sm">
              <div className="flex items-center space-x-4">
                {/* Icon */}
                <div className={`w-10 h-10 bg-gradient-to-br ${gradient} rounded-lg flex items-center justify-center flex-shrink-0`}>
                  <IconComponent className="w-5 h-5 text-white" />
                </div>

                {/* Document Info */}
                <div className="flex-1 min-w-0">
                  <h4 className="font-medium text-gray-900 truncate">{document.name}</h4>
                  <div className="flex items-center space-x-4 text-xs text-gray-500 mt-1">
                    <span>{formatFileSize(document.size)}</span>
                    <span>{document.author.name}</span>
                    <span>{document.updatedAt.toLocaleDateString()}</span>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center space-x-1 flex-shrink-0">
                  <Button variant="ghost" size="sm" onClick={() => onView?.(document.id)}>
                    <Eye className="w-3 h-3" />
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => onDownload?.(document.id)}>
                    <Download className="w-3 h-3" />
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => onShare?.(document.id)}>
                    <Share className="w-3 h-3" />
                  </Button>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        );
      })}
    </div>
  );
}; 