import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { ToastProvider } from "@/components/ui/Toast";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import "./globals.css";
import Link from 'next/link';

const inter = Inter({ 
  subsets: ["latin"],
  variable: "--font-inter",
  display: 'swap',
});

export const metadata: Metadata = {
  title: "Code Hero - AI Agent Platform",
  description: "Beautiful AI-powered development workspace with multi-agent coordination",
  keywords: ["AI", "agents", "development", "FastAPI", "Next.js", "LangChain"],
  authors: [{ name: "Code Hero Team" }],
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-theme="light" className={inter.variable}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className={`${inter.className} antialiased min-h-screen bg-base-100`}>
        <ErrorBoundary>
          <ToastProvider>
            {/* Navigation */}
            <div className="navbar bg-base-100 shadow-lg">
              <div className="navbar-start">
                <Link href="/" className="btn btn-ghost text-xl">
                  🦸 Code Hero
                </Link>
              </div>
              <div className="navbar-center hidden lg:flex">
                <ul className="menu menu-horizontal px-1">
                  <li>
                    <Link href="/" className="btn btn-ghost">
                      Home
                    </Link>
                  </li>
                  <li>
                    <Link href="/chat" className="btn btn-ghost">
                      Chat
                    </Link>
                  </li>
                  <li>
                    <Link href="/agents" className="btn btn-ghost">
                      Agents
                    </Link>
                  </li>
                </ul>
              </div>
              <div className="navbar-end">
                <div className="dropdown dropdown-end lg:hidden">
                  <div tabIndex={0} role="button" className="btn btn-ghost">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h8m-8 6h16" />
                    </svg>
                  </div>
                  <ul tabIndex={0} className="menu menu-sm dropdown-content mt-3 z-[1] p-2 shadow bg-base-100 rounded-box w-52">
                    <li><Link href="/">Home</Link></li>
                    <li><Link href="/chat">Chat</Link></li>
                    <li><Link href="/agents">Agents</Link></li>
                  </ul>
                </div>
              </div>
            </div>
            
            {/* Main content */}
            <main>
              {children}
            </main>
          </ToastProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}
