import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import UploadPanel from './components/UploadPanel';
import ChatWindow from './components/ChatWindow';
import './App.css';

// Dynamically fetch VITE_API_URL from environment or fallback
let rawApiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
if (rawApiUrl && !rawApiUrl.startsWith('http://') && !rawApiUrl.startsWith('https://')) {
  rawApiUrl = `https://${rawApiUrl}`;
}
const API_URL = rawApiUrl;

function App() {
  const [activeCollection, setActiveCollection] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Model & Provider states
  const [providers, setProviders] = useState([]);
  const [options, setOptions] = useState({});
  const [selectedProvider, setSelectedProvider] = useState('');
  const [selectedLlmId, setSelectedLlmId] = useState('');
  const [selectedEmbeddingId, setSelectedEmbeddingId] = useState('');

  // Session states
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);

  // Lifted collection/stats states
  const [collections, setCollections] = useState([]);
  const [stats, setStats] = useState({ total_documents: 0, total_chunks: 0, total_folders: 0 });
  const [collectionsLoading, setCollectionsLoading] = useState(false);

  // Fetch model options and session list on mount
  useEffect(() => {
    const fetchModelOptions = async () => {
      try {
        const res = await fetch(`${API_URL}/api/models/options`);
        if (res.ok) {
          const data = await res.json();
          setProviders(data.providers || []);
          setOptions(data.options || {});
          setSelectedProvider(data.default_provider || '');
          setSelectedLlmId(data.default_llm || '');
          setSelectedEmbeddingId(data.default_embedding || '');
        }
      } catch (err) {
        console.error("Failed to fetch model options", err);
      }
    };

    fetchModelOptions();
    fetchSessionsAndInit();
  }, []);

  // Fetch collections and stats whenever selectedEmbeddingId or refreshTrigger changes
  useEffect(() => {
    const fetchCollectionsAndStats = async () => {
      setCollectionsLoading(true);
      try {
        const url = selectedEmbeddingId 
          ? `${API_URL}/api/collections?embedding_id=${encodeURIComponent(selectedEmbeddingId)}`
          : `${API_URL}/api/collections`;
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          setCollections(data.collections || []);
          setStats(data.statistics || { total_documents: 0, total_chunks: 0, total_folders: 0 });
        }
      } catch (err) {
        console.error("Failed to fetch collections and stats", err);
      } finally {
        setCollectionsLoading(false);
      }
    };

    fetchCollectionsAndStats();
  }, [refreshTrigger, selectedEmbeddingId]);

  const fetchSessions = async (activeId) => {
    try {
      const activeIdToUse = activeId !== undefined ? activeId : currentSessionId;
      const url = activeIdToUse 
        ? `${API_URL}/api/sessions?active_session_id=${encodeURIComponent(activeIdToUse)}`
        : `${API_URL}/api/sessions`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
        return data;
      }
    } catch (err) {
      console.error("Failed to fetch sessions", err);
    }
    return [];
  };

  // Reload session list when currentSessionId or refreshTrigger changes to trigger empty session cleanup
  useEffect(() => {
    if (currentSessionId) {
      fetchSessions(currentSessionId);
    }
  }, [currentSessionId, refreshTrigger]);

  // Reset active collection if it is no longer in the filtered collections list (prevents dimension mismatch error)
  useEffect(() => {
    if (activeCollection && collections && collections.length > 0) {
      if (!collections.includes(activeCollection)) {
        setActiveCollection(null);
      }
    } else if (activeCollection && (!collections || collections.length === 0)) {
      setActiveCollection(null);
    }
  }, [collections, activeCollection]);

  const fetchSessionsAndInit = async () => {
    const loadedSessions = await fetchSessions();
    if (loadedSessions && loadedSessions.length > 0) {
      // Pick the most recent session
      setCurrentSessionId(loadedSessions[0].session_id);
    } else {
      // Create a default session if none exist
      await createNewSession();
    }
  };

  const createNewSession = async () => {
    try {
      const res = await fetch(`${API_URL}/api/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_name: 'react_user' })
      });
      if (res.ok) {
        const data = await res.json();
        // Pass the new session ID to clean up any empty sessions
        await fetchSessions(data.session_id);
        setCurrentSessionId(data.session_id);
        return data.session_id;
      }
    } catch (err) {
      console.error("Failed to create new session", err);
    }
    return null;
  };

  const deleteSession = async (sessionId) => {
    if (!window.confirm("Are you sure you want to delete this conversation?")) return;
    try {
      const res = await fetch(`${API_URL}/api/sessions/${sessionId}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        const remaining = await fetchSessions();
        if (currentSessionId === sessionId) {
          if (remaining && remaining.length > 0) {
            setCurrentSessionId(remaining[0].session_id);
          } else {
            await createNewSession();
          }
        }
      }
    } catch (err) {
      console.error("Failed to delete session", err);
    }
  };

  // Align embedding + LLM with current provider
  useEffect(() => {
    if (!selectedProvider || !options || !options[selectedProvider]) return;
    const providerOptions = options[selectedProvider];
    
    // Align embeddings
    const embeds = providerOptions.embeddings || [];
    if (embeds.length > 0) {
      const exists = embeds.some(e => e.id === selectedEmbeddingId);
      if (!exists) {
        setSelectedEmbeddingId(embeds[0].id);
      }
    }
    
    // Align LLMs
    const llms = providerOptions.llms || [];
    if (llms.length > 0) {
      const exists = llms.some(l => l.id === selectedLlmId);
      if (!exists) {
        setSelectedLlmId(llms[0].id);
      }
    }
  }, [selectedProvider, options]);

  const handleUploadSuccess = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className="app-layout">
      {/* Sidebar - lists folders, stats, model configs, and previous chats */}
      <Sidebar 
        apiUrl={API_URL} 
        activeCollection={activeCollection} 
        setActiveCollection={setActiveCollection}
        refreshTrigger={refreshTrigger}
        setRefreshTrigger={setRefreshTrigger}
        
        // Model configs
        selectedProvider={selectedProvider}
        setSelectedProvider={setSelectedProvider}
        selectedLlmId={selectedLlmId}
        setSelectedLlmId={setSelectedLlmId}
        selectedEmbeddingId={selectedEmbeddingId}
        setSelectedEmbeddingId={setSelectedEmbeddingId}
        providers={providers}
        options={options}
        
        // Sessions
        currentSessionId={currentSessionId}
        setCurrentSessionId={setCurrentSessionId}
        sessions={sessions}
        createNewSession={createNewSession}
        deleteSession={deleteSession}

        // Lifted collections/stats
        collections={collections}
        stats={stats}
        collectionsLoading={collectionsLoading}
      />
      
      {/* Main workspace area */}
      <main className="workspace-container">
        <div className="workspace-chat">
          <ChatWindow 
            apiUrl={API_URL} 
            activeCollection={activeCollection} 
            currentSessionId={currentSessionId}
            selectedLlmId={selectedLlmId}
            selectedEmbeddingId={selectedEmbeddingId}
            onMessageSent={fetchSessions} // refresh sessions to capture preview updates
          />
        </div>
        
        <div className="workspace-panel">
          <UploadPanel 
            apiUrl={API_URL} 
            onUploadSuccess={handleUploadSuccess} 
            collections={collections}
            selectedEmbeddingId={selectedEmbeddingId}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
