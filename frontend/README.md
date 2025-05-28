# Code Hero Frontend

A modern, responsive frontend for the Code Hero AI agent platform built with Next.js 14, TypeScript, and Tailwind CSS.

## Features

- 🤖 **Real-time Agent Management** - Monitor and interact with your AI agents
- 💬 **Advanced Chat Interface** - Seamless communication with AI agents
- 📊 **Comprehensive Dashboard** - System health, metrics, and activity monitoring
- ⚙️ **Settings Management** - User preferences, security, and system configuration
- 🎨 **Modern UI/UX** - Apple-inspired design with glass morphism effects
- 📱 **Responsive Design** - Works perfectly on desktop, tablet, and mobile

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS + DaisyUI
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **State Management**: React Hooks
- **API Client**: Custom fetch-based client

## Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Code Hero backend running (see backend README)

### Installation

1. **Clone and navigate to frontend**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Configure environment**:
   ```bash
   # Create environment file
   cp .env.example .env.local
   
   # Edit .env.local and set your backend URL
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

4. **Start development server**:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

5. **Open in browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

## Environment Configuration

Create a `.env.local` file in the frontend directory:

```env
# Backend API URL - Update this to match your backend server
NEXT_PUBLIC_API_URL=http://localhost:8000

# Development settings
NODE_ENV=development

# Optional: Enable debug logging
NEXT_PUBLIC_DEBUG=true

# Optional: API timeout settings
NEXT_PUBLIC_API_TIMEOUT=30000
```

## Backend Connection

The frontend automatically connects to your Code Hero backend through the API client. Make sure:

1. **Backend is running** on the configured URL (default: `http://localhost:8000`)
2. **CORS is enabled** in your backend for the frontend domain
3. **API endpoints** are accessible and responding

### API Endpoints Used

- `GET /health` - System health check
- `POST /api/chat/` - Send chat messages
- `GET /api/chat/{conversation_id}` - Get chat history
- `GET /api/agents/` - List all agents
- `GET /api/agents/{agent_id}` - Get specific agent
- `POST /api/agents/{agent_id}/interact` - Interact with agent
- `GET /api/agents/{agent_id}/history` - Get agent history
- `GET /api/agents/statistics/overview` - Get agent statistics
- `POST /multi-agent/coordinate` - Multi-agent task coordination

## Project Structure

```
frontend/
├── src/
│   ├── app/                 # Next.js App Router pages
│   │   ├── dashboard/       # Dashboard page
│   │   ├── agents/          # Agents management page
│   │   ├── chat/            # Chat interface page
│   │   ├── settings/        # Settings page
│   │   └── layout.tsx       # Root layout
│   ├── components/          # React components
│   │   ├── layout/          # Layout components (Header, Sidebar, Footer)
│   │   ├── ui/              # UI components (Button, Input, Cards, etc.)
│   │   └── chat/            # Chat-specific components
│   ├── hooks/               # Custom React hooks
│   │   ├── useAgents.ts     # Agent management hook
│   │   ├── useChat.ts       # Chat functionality hook
│   │   └── useHealth.ts     # System health hook
│   ├── utils/               # Utility functions
│   │   └── api.ts           # API client and types
│   └── styles/              # Global styles
├── public/                  # Static assets
└── package.json            # Dependencies and scripts
```

## Key Components

### Layout Components
- **Header**: Navigation, user menu, system status
- **Sidebar**: Main navigation with collapsible design
- **Footer**: Links, social media, system information

### UI Components
- **GlassCard**: Apple-inspired glass morphism cards
- **Button**: Consistent button styling with variants
- **LoadingSpinner**: Multiple loading states and animations
- **ErrorBoundary**: Comprehensive error handling
- **Toast**: Notification system with multiple types

### Feature Components
- **ChatInterface**: Real-time chat with AI agents
- **AgentCard**: Agent status and management
- **MetricCard**: Dashboard metrics display
- **SettingsPanel**: Configuration management

## Development

### Available Scripts

```bash
# Development server
npm run dev

# Production build
npm run build

# Start production server
npm start

# Type checking
npm run type-check

# Linting
npm run lint

# Linting with auto-fix
npm run lint:fix
```

### Code Style

The project follows these conventions:

- **TypeScript**: Strict mode enabled
- **ESLint**: Next.js recommended rules
- **Prettier**: Consistent code formatting
- **Component Structure**: Functional components with hooks
- **File Naming**: kebab-case for files, PascalCase for components
- **Import Order**: External libraries, internal modules, relative imports

### Adding New Features

1. **Create component** in appropriate directory
2. **Add TypeScript interfaces** for props and data
3. **Implement responsive design** with Tailwind CSS
4. **Add error handling** and loading states
5. **Update API client** if new endpoints needed
6. **Add tests** for critical functionality

## Deployment

### Vercel (Recommended)

1. **Connect repository** to Vercel
2. **Set environment variables** in Vercel dashboard
3. **Deploy** automatically on push to main branch

### Manual Deployment

```bash
# Build for production
npm run build

# Start production server
npm start
```

## Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Check if backend is running on configured URL
   - Verify CORS settings in backend
   - Check network connectivity

2. **Build Errors**
   - Run `npm run type-check` to identify TypeScript issues
   - Check for missing dependencies
   - Verify environment variables

3. **Styling Issues**
   - Clear browser cache
   - Check Tailwind CSS configuration
   - Verify DaisyUI theme settings

### Debug Mode

Enable debug logging by setting `NEXT_PUBLIC_DEBUG=true` in your environment file.

## Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

## License

This project is part of the Code Hero AI platform. See the main repository for license information.

## Support

For support and questions:
- Check the [main repository](../README.md) for general information
- Review the [backend documentation](../src/README.md) for API details
- Open an issue for bugs or feature requests
