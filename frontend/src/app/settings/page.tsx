'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Layout } from '@/components/layout/Layout';
import { GlassCard } from '@/components/ui/GlassCard';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useToastActions } from '@/components/ui/Toast';
import { 
  User, 
  Bell, 
  Shield, 
  Palette, 
  Database,
  Key,
  Globe,
  Monitor,
  Moon,
  Sun,
  Volume2,
  Mail,
  Smartphone,
  Eye,
  EyeOff,
  Save,
  RefreshCw,
  Trash2,
  Download,
  Upload,
  Settings as SettingsIcon
} from 'lucide-react';
import { clsx } from 'clsx';

interface SettingsSection {
  id: string;
  title: string;
  icon: React.ReactNode;
  description: string;
}

export default function SettingsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeSection, setActiveSection] = useState('profile');
  const [showPassword, setShowPassword] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [notifications, setNotifications] = useState({
    email: true,
    push: true,
    desktop: false,
    sound: true
  });
  const [profile, setProfile] = useState({
    name: 'John Doe',
    email: 'john.doe@example.com',
    company: 'Code Hero Inc.',
    role: 'Senior Developer',
    bio: 'Passionate developer building amazing applications with AI assistance.',
    avatar: ''
  });
  const toast = useToastActions();

  const settingsSections: SettingsSection[] = [
    {
      id: 'profile',
      title: 'Profile',
      icon: <User className="w-5 h-5" />,
      description: 'Manage your personal information and preferences'
    },
    {
      id: 'notifications',
      title: 'Notifications',
      icon: <Bell className="w-5 h-5" />,
      description: 'Configure how you receive updates and alerts'
    },
    {
      id: 'security',
      title: 'Security',
      icon: <Shield className="w-5 h-5" />,
      description: 'Manage your account security and privacy settings'
    },
    {
      id: 'appearance',
      title: 'Appearance',
      icon: <Palette className="w-5 h-5" />,
      description: 'Customize the look and feel of your workspace'
    },
    {
      id: 'integrations',
      title: 'Integrations',
      icon: <Database className="w-5 h-5" />,
      description: 'Connect with external services and APIs'
    },
    {
      id: 'advanced',
      title: 'Advanced',
      icon: <SettingsIcon className="w-5 h-5" />,
      description: 'Advanced configuration and developer settings'
    }
  ];

  const handleSaveProfile = () => {
    toast.success('Profile updated', 'Your profile information has been saved successfully');
  };

  const handleSaveNotifications = () => {
    toast.success('Notifications updated', 'Your notification preferences have been saved');
  };

  const handleExportData = () => {
    toast.info('Export started', 'Your data export will be ready shortly');
  };

  const handleDeleteAccount = () => {
    if (confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
      toast.error('Account deletion requested', 'Please check your email to confirm');
    }
  };

  const renderProfileSection = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Personal Information</h3>
        <p className="text-gray-600 text-sm mb-6">
          Update your personal details and how others see you on Code Hero.
        </p>
      </div>

      {/* Avatar */}
      <div className="flex items-center space-x-6">
        <div className="w-20 h-20 bg-gradient-to-br from-system-blue to-system-indigo rounded-2xl flex items-center justify-center text-white shadow-lg">
          <User className="w-10 h-10" />
        </div>
        <div className="space-y-2">
          <Button variant="ghost" className="text-system-blue">
            <Upload className="w-4 h-4 mr-2" />
            Upload new photo
          </Button>
          <p className="text-xs text-gray-500">
            JPG, PNG or GIF. Max size 2MB.
          </p>
        </div>
      </div>

      {/* Form Fields */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Input
          label="Full Name"
          value={profile.name}
          onChange={(e) => setProfile({ ...profile, name: e.target.value })}
          placeholder="Enter your full name"
        />
        <Input
          label="Email Address"
          type="email"
          value={profile.email}
          onChange={(e) => setProfile({ ...profile, email: e.target.value })}
          placeholder="Enter your email"
        />
        <Input
          label="Company"
          value={profile.company}
          onChange={(e) => setProfile({ ...profile, company: e.target.value })}
          placeholder="Enter your company"
        />
        <Input
          label="Role"
          value={profile.role}
          onChange={(e) => setProfile({ ...profile, role: e.target.value })}
          placeholder="Enter your role"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Bio
        </label>
        <textarea
          value={profile.bio}
          onChange={(e) => setProfile({ ...profile, bio: e.target.value })}
          rows={4}
          className={clsx(
            'w-full px-4 py-3 rounded-xl border border-gray-200',
            'bg-white/50 backdrop-blur-sm resize-none',
            'focus:outline-none focus:ring-2 focus:ring-system-blue/20 focus:border-system-blue',
            'transition-all duration-200'
          )}
          placeholder="Tell us about yourself..."
        />
      </div>

      <div className="flex justify-end">
        <Button
          onClick={handleSaveProfile}
          className="bg-gradient-to-r from-system-blue to-system-indigo text-white flex items-center space-x-2"
        >
          <Save className="w-4 h-4" />
          <span>Save Changes</span>
        </Button>
      </div>
    </div>
  );

  const renderNotificationsSection = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Notification Preferences</h3>
        <p className="text-gray-600 text-sm mb-6">
          Choose how you want to be notified about important updates and activities.
        </p>
      </div>

      <div className="space-y-4">
        {[
          { key: 'email', icon: Mail, title: 'Email Notifications', description: 'Receive updates via email' },
          { key: 'push', icon: Smartphone, title: 'Push Notifications', description: 'Browser push notifications' },
          { key: 'desktop', icon: Monitor, title: 'Desktop Notifications', description: 'System desktop notifications' },
          { key: 'sound', icon: Volume2, title: 'Sound Alerts', description: 'Play sounds for notifications' }
        ].map(({ key, icon: Icon, title, description }) => (
          <div key={key} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center shadow-sm">
                <Icon className="w-5 h-5 text-gray-600" />
              </div>
              <div>
                <h4 className="font-medium text-gray-900">{title}</h4>
                <p className="text-sm text-gray-600">{description}</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={notifications[key as keyof typeof notifications]}
                onChange={(e) => setNotifications({ ...notifications, [key]: e.target.checked })}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-system-blue/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-system-blue"></div>
            </label>
          </div>
        ))}
      </div>

      <div className="flex justify-end">
        <Button
          onClick={handleSaveNotifications}
          className="bg-gradient-to-r from-system-blue to-system-indigo text-white flex items-center space-x-2"
        >
          <Save className="w-4 h-4" />
          <span>Save Preferences</span>
        </Button>
      </div>
    </div>
  );

  const renderSecuritySection = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Security & Privacy</h3>
        <p className="text-gray-600 text-sm mb-6">
          Manage your account security and privacy settings.
        </p>
      </div>

      {/* Password Change */}
      <div className="space-y-4">
        <h4 className="font-medium text-gray-900">Change Password</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="relative">
            <Input
              label="Current Password"
              type={showPassword ? 'text' : 'password'}
              placeholder="Enter current password"
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-9 text-gray-400 hover:text-gray-600"
            >
              {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
            </button>
          </div>
          <Input
            label="New Password"
            type="password"
            placeholder="Enter new password"
          />
        </div>
      </div>

      {/* Two-Factor Authentication */}
      <div className="p-4 bg-gray-50 rounded-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-system-green/10 rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5 text-system-green" />
            </div>
            <div>
              <h4 className="font-medium text-gray-900">Two-Factor Authentication</h4>
              <p className="text-sm text-gray-600">Add an extra layer of security</p>
            </div>
          </div>
          <Button variant="ghost" className="text-system-blue">
            Enable
          </Button>
        </div>
      </div>

      {/* API Keys */}
      <div className="space-y-4">
        <h4 className="font-medium text-gray-900">API Keys</h4>
        <div className="p-4 bg-gray-50 rounded-xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Key className="w-5 h-5 text-gray-600" />
              <div>
                <p className="font-medium text-gray-900">Personal Access Token</p>
                <p className="text-sm text-gray-600">Created 2 days ago</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button size="sm" variant="ghost">
                <RefreshCw className="w-4 h-4" />
              </Button>
              <Button size="sm" variant="ghost" className="text-system-red">
                <Trash2 className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderAppearanceSection = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Appearance</h3>
        <p className="text-gray-600 text-sm mb-6">
          Customize how Code Hero looks and feels.
        </p>
      </div>

      {/* Theme Selection */}
      <div className="space-y-4">
        <h4 className="font-medium text-gray-900">Theme</h4>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            { id: 'light', name: 'Light', icon: Sun, active: !isDarkMode },
            { id: 'dark', name: 'Dark', icon: Moon, active: isDarkMode },
            { id: 'system', name: 'System', icon: Monitor, active: false }
          ].map((theme) => (
            <button
              key={theme.id}
              onClick={() => setIsDarkMode(theme.id === 'dark')}
              className={clsx(
                'p-4 rounded-xl border-2 transition-all duration-200',
                theme.active 
                  ? 'border-system-blue bg-system-blue/5' 
                  : 'border-gray-200 hover:border-gray-300'
              )}
            >
              <div className="flex flex-col items-center space-y-2">
                <theme.icon className={clsx(
                  'w-6 h-6',
                  theme.active ? 'text-system-blue' : 'text-gray-600'
                )} />
                <span className={clsx(
                  'font-medium',
                  theme.active ? 'text-system-blue' : 'text-gray-900'
                )}>
                  {theme.name}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Language */}
      <div className="space-y-4">
        <h4 className="font-medium text-gray-900">Language</h4>
        <div className="relative">
          <Globe className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <select className={clsx(
            'w-full pl-10 pr-4 py-3 rounded-xl border border-gray-200',
            'bg-white/50 backdrop-blur-sm appearance-none',
            'focus:outline-none focus:ring-2 focus:ring-system-blue/20 focus:border-system-blue',
            'transition-all duration-200'
          )}>
            <option>English (US)</option>
            <option>English (UK)</option>
            <option>Spanish</option>
            <option>French</option>
            <option>German</option>
          </select>
        </div>
      </div>
    </div>
  );

  const renderAdvancedSection = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Advanced Settings</h3>
        <p className="text-gray-600 text-sm mb-6">
          Advanced configuration options and data management.
        </p>
      </div>

      {/* Data Export */}
      <div className="p-4 bg-gray-50 rounded-xl">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-medium text-gray-900">Export Data</h4>
            <p className="text-sm text-gray-600">Download all your data in JSON format</p>
          </div>
          <Button
            onClick={handleExportData}
            variant="ghost"
            className="text-system-blue flex items-center space-x-2"
          >
            <Download className="w-4 h-4" />
            <span>Export</span>
          </Button>
        </div>
      </div>

      {/* Danger Zone */}
      <div className="p-6 border-2 border-system-red/20 rounded-xl bg-system-red/5">
        <h4 className="font-medium text-system-red mb-2">Danger Zone</h4>
        <p className="text-sm text-gray-600 mb-4">
          These actions are irreversible. Please proceed with caution.
        </p>
        <Button
          onClick={handleDeleteAccount}
          variant="ghost"
          className="text-system-red hover:bg-system-red/10 flex items-center space-x-2"
        >
          <Trash2 className="w-4 h-4" />
          <span>Delete Account</span>
        </Button>
      </div>
    </div>
  );

  const renderContent = () => {
    switch (activeSection) {
      case 'profile':
        return renderProfileSection();
      case 'notifications':
        return renderNotificationsSection();
      case 'security':
        return renderSecuritySection();
      case 'appearance':
        return renderAppearanceSection();
      case 'integrations':
        return (
          <div className="text-center py-12">
            <Database className="w-16 h-16 mx-auto text-gray-400 mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Integrations</h3>
            <p className="text-gray-600">Integration settings coming soon...</p>
          </div>
        );
      case 'advanced':
        return renderAdvancedSection();
      default:
        return renderProfileSection();
    }
  };

  return (
    <Layout
      sidebarCollapsed={sidebarCollapsed}
      onSidebarToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
    >
      <div className="space-y-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1 className="text-fluid-3xl font-bold text-gray-900">Settings</h1>
          <p className="text-gray-600 mt-2">
            Manage your account settings and preferences
          </p>
        </motion.div>

        {/* Settings Content */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 lg:grid-cols-4 gap-8"
        >
          {/* Settings Navigation */}
          <div className="lg:col-span-1">
            <GlassCard className="p-4">
              <nav className="space-y-2">
                {settingsSections.map((section) => (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={clsx(
                      'w-full flex items-center space-x-3 px-3 py-3 rounded-xl transition-all duration-200 text-left',
                      activeSection === section.id
                        ? 'bg-system-blue/10 text-system-blue'
                        : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                    )}
                  >
                    {section.icon}
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">{section.title}</p>
                      <p className="text-xs opacity-75 truncate">{section.description}</p>
                    </div>
                  </button>
                ))}
              </nav>
            </GlassCard>
          </div>

          {/* Settings Content */}
          <div className="lg:col-span-3">
            <GlassCard className="p-8">
              <motion.div
                key={activeSection}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3 }}
              >
                {renderContent()}
              </motion.div>
            </GlassCard>
          </div>
        </motion.div>
      </div>
    </Layout>
  );
} 