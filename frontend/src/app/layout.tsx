import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ 
  subsets: ["latin"],
  variable: "--font-inter",
  display: 'swap',
});

export const metadata: Metadata = {
  title: "Code Hero - AI Agent Platform",
  description: "Beautiful AI agent coordination platform inspired by Imagica.ai with multi-agent workflows, real-time collaboration, and stunning visual design.",
  keywords: ["AI", "agents", "automation", "code", "development", "LangGraph", "FastAPI", "Next.js"],
  authors: [{ name: "Code Hero Team" }],
  viewport: "width=device-width, initial-scale=1",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-theme="imagica" className={inter.variable}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className={`${inter.className} antialiased bg-mesh-light min-h-screen`}>
        <div className="relative">
          {/* Background gradient mesh */}
          <div className="fixed inset-0 bg-mesh-light -z-10" />
          
          {/* Main content */}
          <div className="relative z-10">
            {children}
          </div>
        </div>
      </body>
    </html>
  );
}
