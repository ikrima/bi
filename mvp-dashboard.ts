// app/page.tsx
// The entire dashboard in one file. Ship it.

'use client';

import { useState, useEffect } from 'react';

// Mock data for now - replace with API calls
const mockData = {
  overview: {
    totalMessages: 1247,
    uniquePlayers: 89,
    sentiment: 72,
    urgentIssues: 3
  },
  themes: [
    { theme: 'weapon + balance + nerf', sentiment: 'Negative', count: 234, sample: 'The sniper rifle is completely broken...' },
    { theme: 'map + design + love', sentiment: 'Positive', count: 189, sample: 'The new map is absolutely amazing...' },
    { theme: 'bug + crash + server', sentiment: 'Negative', count: 156, sample: 'Game crashes every time I try to...' },
    { theme: 'event + rewards + fun', sentiment: 'Positive', count: 134, sample: 'This event is so much fun...' },
    { theme: 'tutorial + confusing + help', sentiment: 'Neutral', count: 98, sample: 'The tutorial doesnt explain how to...' }
  ],
  urgentIssues: [
    { author: 'PlayerOne', message: 'Game crashes on startup after update', time: '2 hours ago' },
    { author: 'xXGamerXx', message: 'Lost all my progress, this is unacceptable', time: '3 hours ago' },
    { author: 'CasualPlayer', message: 'Cant connect to servers, error code 2001', time: '5 hours ago' }
  ],
  sentimentHistory: [
    { day: 'Mon', score: 68 },
    { day: 'Tue', score: 72 },
    { day: 'Wed', score: 71 },
    { day: 'Thu', score: 65 },
    { day: 'Fri', score: 69 },
    { day: 'Sat', score: 74 },
    { day: 'Sun', score: 72 }
  ]
};

export default function Dashboard() {
  const [data, setData] = useState(mockData);
  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Fetch real data (implement this)
  const fetchData = async () => {
    setLoading(true);
    try {
      // const response = await fetch('/api/digest');
      // const newData = await response.json();
      // setData(newData);
      
      // For now, just update the timestamp
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch data:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    // Fetch data on mount
    fetchData();
    
    // Refresh every 5 minutes
    const interval = setInterval(fetchData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const getSentimentColor = (score: number) => {
    if (score >= 70) return '#10b981'; // green
    if (score >= 40) return '#f59e0b'; // yellow
    return '#ef4444'; // red
  };

  const getSentimentEmoji = (sentiment: string) => {
    if (sentiment === 'Positive') return '😊';
    if (sentiment === 'Negative') return '😟';
    return '😐';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-semibold text-gray-900">
                Player Intelligence
              </h1>
              <p className="text-sm text-gray-500">
                Last updated: {lastUpdate.toLocaleTimeString()}
              </p>
            </div>
            <button
              onClick={fetchData}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-sm font-medium text-gray-500">Total Messages</div>
            <div className="mt-2 text-3xl font-semibold text-gray-900">
              {data.overview.totalMessages.toLocaleString()}
            </div>
            <div className="mt-1 text-sm text-green-600">↑ 12% from yesterday</div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-sm font-medium text-gray-500">Active Players</div>
            <div className="mt-2 text-3xl font-semibold text-gray-900">
              {data.overview.uniquePlayers}
            </div>
            <div className="mt-1 text-sm text-gray-600">In last 24 hours</div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-sm font-medium text-gray-500">Sentiment Score</div>
            <div className="mt-2 text-3xl font-semibold" style={{ color: getSentimentColor(data.overview.sentiment) }}>
              {data.overview.sentiment}
            </div>
            <div className="mt-1 text-sm text-gray-600">Out of 100</div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-sm font-medium text-gray-500">Urgent Issues</div>
            <div className="mt-2 text-3xl font-semibold text-red-600">
              {data.overview.urgentIssues}
            </div>
            <div className="mt-1 text-sm text-red-600">Need attention</div>
          </div>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Main Themes */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b">
              <h2 className="text-lg font-semibold text-gray-900">
                🎯 Main Discussion Topics
              </h2>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                {data.themes.map((theme, index) => (
                  <div key={index} className="border-l-4 border-blue-500 pl-4 py-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium text-gray-900">
                          {theme.theme}
                        </span>
                        <span className="text-lg">{getSentimentEmoji(theme.sentiment)}</span>
                      </div>
                      <span className="text-sm text-gray-500">
                        {theme.count} messages
                      </span>
                    </div>
                    <p className="mt-1 text-sm text-gray-600 italic">
                      "{theme.sample}..."
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Urgent Issues */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b bg-red-50">
              <h2 className="text-lg font-semibold text-red-900">
                ⚠️ Urgent Issues
              </h2>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                {data.urgentIssues.map((issue, index) => (
                  <div key={index} className="bg-red-50 rounded-lg p-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-medium text-gray-900">{issue.author}</div>
                        <p className="mt-1 text-sm text-gray-700">"{issue.message}"</p>
                      </div>
                      <span className="text-xs text-gray-500 whitespace-nowrap ml-4">
                        {issue.time}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <button className="mt-4 w-full px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700">
                View All Issues
              </button>
            </div>
          </div>
        </div>

        {/* Sentiment Trend */}
        <div className="mt-8 bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b">
            <h2 className="text-lg font-semibold text-gray-900">
              📈 7-Day Sentiment Trend
            </h2>
          </div>
          <div className="p-6">
            <div className="flex items-end justify-between h-48">
              {data.sentimentHistory.map((day, index) => (
                <div key={index} className="flex flex-col items-center flex-1">
                  <div className="w-full max-w-12 bg-gray-200 rounded-t-lg relative">
                    <div
                      className="absolute bottom-0 w-full rounded-t-lg transition-all duration-300"
                      style={{
                        height: `${(day.score / 100) * 192}px`,
                        backgroundColor: getSentimentColor(day.score)
                      }}
                    />
                  </div>
                  <div className="mt-2 text-xs text-gray-600">{day.day}</div>
                  <div className="text-xs font-medium">{day.score}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="mt-8 bg-gradient-to-r from-blue-600 to-blue-700 rounded-lg shadow-lg">
          <div className="px-8 py-12 text-center">
            <h3 className="text-2xl font-bold text-white mb-4">
              You just saved 2 hours of Discord scrolling
            </h3>
            <p className="text-blue-100 mb-8">
              Get these insights delivered to your inbox every morning at 8 AM
            </p>
            <div className="flex justify-center space-x-4">
              <button className="px-6 py-3 bg-white text-blue-600 rounded-lg font-semibold hover:bg-gray-100">
                Configure Digest
              </button>
              <button className="px-6 py-3 bg-blue-800 text-white rounded-lg font-semibold hover:bg-blue-900">
                Upgrade to Pro ($499/mo)
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}