import React, { useEffect, useState } from 'react';
import './Sidebar.css';

export default function Sidebar({ 
  apiUrl, 
  activeCollection, 
  setActiveCollection, 
  refreshTrigger, 
  setRefreshTrigger,
  
  // Model configs
  selectedProvider,
  setSelectedProvider,
  selectedLlmId,
  setSelectedLlmId,
  selectedEmbeddingId,
  setSelectedEmbeddingId,
  providers,
  options,
  
  // Sessions
  currentSessionId,
  setCurrentSessionId,
  sessions,
  createNewSession,
  deleteSession,

  // Lifted state
  collections,
  stats,
  collectionsLoading: loading
}) {
  const [expandedFolder, setExpandedFolder] = useState(null);
  const [folderFiles, setFolderFiles] = useState({});
  const [filesLoading, setFilesLoading] = useState({});
  const [folderStats, setFolderStats] = useState({});
  const [statsLoading, setStatsLoading] = useState({});

  // Inventory Report States
  const [showInventory, setShowInventory] = useState(false);
  const [inventoryReport, setInventoryReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);

  const fetchInventoryReport = async () => {
    setReportLoading(true);
    try {
      const res = await fetch(`${apiUrl}/api/collections/inventory/report`);
      if (res.ok) {
        const data = await res.json();
        setInventoryReport(data);
      }
    } catch (err) {
      console.error("Failed to fetch inventory report", err);
    } finally {
      setReportLoading(false);
    }
  };

  // Fetch files and stats for expanded folder automatically
  useEffect(() => {
    if (expandedFolder) {
      // Fetch files
      const fetchFiles = async () => {
        setFilesLoading(prev => ({ ...prev, [expandedFolder]: true }));
        try {
          const res = await fetch(`${apiUrl}/api/collections/${expandedFolder}/documents`);
          if (res.ok) {
            const data = await res.json();
            setFolderFiles(prev => ({ ...prev, [expandedFolder]: data }));
          }
        } catch (err) {
          console.error(`Failed to fetch documents for ${expandedFolder}`, err);
        } finally {
          setFilesLoading(prev => ({ ...prev, [expandedFolder]: false }));
        }
      };

      // Fetch stats
      const fetchStats = async () => {
        setStatsLoading(prev => ({ ...prev, [expandedFolder]: true }));
        try {
          const res = await fetch(`${apiUrl}/api/collections/${expandedFolder}/statistics`);
          if (res.ok) {
            const data = await res.json();
            setFolderStats(prev => ({ ...prev, [expandedFolder]: data }));
          }
        } catch (err) {
          console.error(`Failed to fetch stats for ${expandedFolder}`, err);
        } finally {
          setStatsLoading(prev => ({ ...prev, [expandedFolder]: false }));
        }
      };

      fetchFiles();
      fetchStats();
    }
  }, [apiUrl, expandedFolder, refreshTrigger]);

  const handleFolderClick = (folderName) => {
    setActiveCollection(folderName);
    if (expandedFolder === folderName) {
      setExpandedFolder(null);
    } else {
      setExpandedFolder(folderName);
    }
  };

  const handleDeleteFolder = async (e, folderName) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete the folder "${folderName}" and all its contents?`)) {
      return;
    }
    try {
      const res = await fetch(`${apiUrl}/api/collections/${folderName}`, { method: 'DELETE' });
      if (res.ok) {
        if (activeCollection === folderName) {
          setActiveCollection(null);
        }
        if (expandedFolder === folderName) {
          setExpandedFolder(null);
        }
        setRefreshTrigger(prev => prev + 1);
      }
    } catch (err) {
      console.error("Failed to delete folder", err);
    }
  };

  const handleDeleteFile = async (e, folderName, docId, fileName) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete "${fileName}"?`)) {
      return;
    }
    try {
      const res = await fetch(`${apiUrl}/api/collections/${folderName}/documents/${docId}`, { method: 'DELETE' });
      if (res.ok) {
        // Refresh aggregate and folder stats/files
        setRefreshTrigger(prev => prev + 1);
      }
    } catch (err) {
      console.error("Failed to delete file", err);
    }
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1 className="brand-title">Eva <span className="brand-glow">Research AI</span></h1>
        <p className="brand-subtitle">LangGraph Corrective RAG</p>
      </div>

      {/* Aggregate Stats Dashboard */}
      <div className="stats-dashboard">
        <div className="stat-card">
          <span className="stat-value">{stats.total_folders}</span>
          <span className="stat-label">Folders</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{stats.total_documents}</span>
          <span className="stat-label">Docs</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{stats.total_chunks}</span>
          <span className="stat-label">Chunks</span>
        </div>
      </div>

      {/* Model Configuration Section */}
      <div className="sidebar-group card-glass">
        <h2 className="group-title">⚙️ Models</h2>
        
        <div className="select-container">
          <label htmlFor="provider-select">Embedding/Answer Provider</label>
          <select
            id="provider-select"
            value={selectedProvider || ''}
            onChange={(e) => setSelectedProvider(e.target.value)}
            className="sidebar-select"
          >
            {providers.map(p => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
        </div>
        
        {selectedProvider && options[selectedProvider]?.embeddings && (
          <div className="select-container">
            <label htmlFor="embedding-select">Embedding Model</label>
            <select
              id="embedding-select"
              value={selectedEmbeddingId || ''}
              onChange={(e) => setSelectedEmbeddingId(e.target.value)}
              className="sidebar-select"
            >
              {options[selectedProvider].embeddings.map(emb => (
                <option key={emb.id} value={emb.id}>{emb.label}</option>
              ))}
            </select>
          </div>
        )}

        {selectedProvider && options[selectedProvider]?.llms && (
          <div className="select-container">
            <label htmlFor="llm-select">Answer Model</label>
            <select
              id="llm-select"
              value={selectedLlmId || ''}
              onChange={(e) => setSelectedLlmId(e.target.value)}
              className="sidebar-select"
            >
              {options[selectedProvider].llms.map(llm => (
                <option key={llm.id} value={llm.id}>{llm.label}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Collections/Folders Section */}
      <div className="collections-section">
        <div className="section-header">
          <h2>📁 Collections</h2>
          {loading && <span className="loader-mini"></span>}
        </div>
        
        <div className="collections-list">
          <div 
            className={`collection-item all-folders ${activeCollection === null ? 'active' : ''}`}
            onClick={() => {
              setActiveCollection(null);
              setExpandedFolder(null);
            }}
          >
            <span className="folder-icon">🔍</span>
            <span className="collection-name">Search All Folders</span>
          </div>

          {collections.map(folder => {
            const isExpanded = expandedFolder === folder;
            const isActive = activeCollection === folder;
            const files = folderFiles[folder] || [];
            
            return (
              <div key={folder} className={`collection-group ${isActive ? 'active-group' : ''}`}>
                <div 
                  className={`collection-item ${isActive ? 'active' : ''}`}
                  onClick={() => handleFolderClick(folder)}
                >
                  <span className="folder-chevron">{isExpanded ? '▼' : '▶'}</span>
                  <span className="folder-icon">📁</span>
                  <span className="collection-name">{folder}</span>
                  <button 
                    className="btn-icon-delete"
                    onClick={(e) => handleDeleteFolder(e, folder)}
                    title="Delete folder"
                  >
                    🗑️
                  </button>
                </div>

                {isExpanded && (
                  <div className="folder-contents fade-in">
                    {/* Folder Specific Stats Dashboard */}
                    {statsLoading[folder] && !folderStats[folder] ? (
                      <div className="folder-stats-loading">
                        <span className="spinner-mini"></span> Loading stats...
                      </div>
                    ) : (
                      folderStats[folder] && (
                        <div className="folder-stats-card">
                          <div className="folder-stats-grid">
                            <div className="folder-stat-mini-card">
                              <span className="folder-stat-label">Files</span>
                              <span className="folder-stat-value">{folderStats[folder].total_files}</span>
                            </div>
                            <div className="folder-stat-mini-card">
                              <span className="folder-stat-label">Chunks</span>
                              <span className="folder-stat-value">{folderStats[folder].total_chunks}</span>
                            </div>
                          </div>
                          {folderStats[folder].doc_types && Object.keys(folderStats[folder].doc_types).length > 0 && (
                            <div className="folder-stats-doctypes">
                              <span className="doctypes-title">Parts Breakdown</span>
                              <div className="doctypes-badges">
                                {Object.entries(folderStats[folder].doc_types).map(([type, count]) => (
                                  <span key={type} className="doctype-badge" title={`${count} chunks of type ${type}`}>
                                    {type}: <strong>{count}</strong>
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    )}

                    {filesLoading[folder] && <div className="loading-text">Loading files...</div>}
                    {!filesLoading[folder] && files.length === 0 && (
                      <div className="empty-text">No files in folder</div>
                    )}
                    {!filesLoading[folder] && files.map(file => (
                      <div key={file.doc_id} className="file-item">
                        <span className="file-icon">📄</span>
                        <div className="file-details">
                          <span className="file-name" title={file.file_name}>{file.file_name}</span>
                          <span className="file-chunks">{file.chunks} chunks</span>
                        </div>
                        <button 
                          className="btn-file-delete"
                          onClick={(e) => handleDeleteFile(e, folder, file.doc_id, file.file_name)}
                          title="Delete file"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Inventory Report Section */}
      <div className="sidebar-group card-glass inventory-section">
        <div 
          className="section-header" 
          onClick={() => setShowInventory(!showInventory)}
          style={{ cursor: 'pointer', userSelect: 'none', marginBottom: '8px' }}
        >
          <h2 className="group-title" style={{ margin: 0 }}>📊 Inventory Report {showInventory ? '▼' : '▶'}</h2>
        </div>
        
        {showInventory && (
          <div className="inventory-contents fade-in">
            <button 
              onClick={fetchInventoryReport} 
              className="btn-generate-report"
              disabled={reportLoading}
            >
              {reportLoading ? 'Generating...' : 'Generate Report'}
            </button>
            
            {inventoryReport && (
              <div className="inventory-results">
                <div className="inventory-summary-grid">
                  <div className="summary-grid-card">
                    <span className="summary-val">{inventoryReport.overall?.total_collections || 0}</span>
                    <span className="summary-lbl">Colls</span>
                  </div>
                  <div className="summary-grid-card">
                    <span className="summary-val">{inventoryReport.overall?.total_files || 0}</span>
                    <span className="summary-lbl">Files</span>
                  </div>
                  <div className="summary-grid-card">
                    <span className="summary-val">{inventoryReport.overall?.total_chunks || 0}</span>
                    <span className="summary-lbl">Chunks</span>
                  </div>
                </div>
                
                <div className="inventory-status-row">
                  <span className="status-badge success" title="Successfully indexed files">
                    ✅ {inventoryReport.overall?.files_success || 0}
                  </span>
                  <span className="status-badge failed" title="Failed ingestion files">
                    ❌ {inventoryReport.overall?.files_failed || 0}
                  </span>
                  <span className="status-badge skipped" title="Skipped image-only or duplicate files">
                    ⏭️ {inventoryReport.overall?.files_skipped || 0}
                  </span>
                </div>
                
                {inventoryReport.collections && Object.keys(inventoryReport.collections).length > 0 && (
                  <div className="inventory-colls-breakdown">
                    <h3>Collections List</h3>
                    <div className="inventory-colls-list">
                      {Object.entries(inventoryReport.collections).map(([name, data]) => (
                        <div key={name} className="inventory-coll-item-row">
                          <span className="coll-name-txt" title={name}>📁 {name}</span>
                          <span className="coll-stats-txt">{data.total_files} f / {data.total_chunks} c</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Chat Sessions / History Section */}
      <div className="sessions-section card-glass">
        <button onClick={createNewSession} className="btn-new-chat" style={{ marginTop: 0 }}>
          <span className="btn-icon">➕</span> New Chat
        </button>

        <div className="section-header">
          <h2>💬 Recent Chats</h2>
        </div>

        <div className="sessions-list">
          {sessions.map(s => {
            const isActive = s.session_id === currentSessionId;
            return (
              <div key={s.session_id} className={`session-item-row ${isActive ? 'active' : ''}`}>
                <div 
                  className="session-item-click-target"
                  onClick={() => setCurrentSessionId(s.session_id)}
                  title={s.preview || "New Conversation"}
                >
                  <span className="chat-icon">{isActive ? '📍' : '💬'}</span>
                  <span className="session-preview">{s.preview || "New Conversation"}</span>
                </div>
                <button 
                  className="btn-session-delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(s.session_id);
                  }}
                  title="Delete chat"
                >
                  🗑️
                </button>
              </div>
            );
          })}
          {sessions.length === 0 && (
            <div className="empty-text">No previous conversations</div>
          )}
        </div>
      </div>
    </aside>
  );
}
