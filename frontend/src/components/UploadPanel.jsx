import React, { useState, useRef, useEffect } from 'react';
import './UploadPanel.css';

export default function UploadPanel({ apiUrl, onUploadSuccess, collections = [], selectedEmbeddingId }) {
  const [useExisting, setUseExisting] = useState(true);
  const [selectedCollection, setSelectedCollection] = useState('');
  const [newCollectionName, setNewCollectionName] = useState('');
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null); // 'success', 'error', 'warning', 'loading'
  const [message, setMessage] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const [totalFilesCount, setTotalFilesCount] = useState(0);
  
  const fileInputRef = useRef(null);

  // Synchronize options when collections list changes
  useEffect(() => {
    if (collections && collections.length > 0) {
      setUseExisting(true);
      if (!collections.includes(selectedCollection)) {
        setSelectedCollection(collections[0]);
      }
    } else {
      setUseExisting(false);
      setSelectedCollection('');
    }
  }, [collections]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndAddFiles(e.dataTransfer.files);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndAddFiles(e.target.files);
    }
  };

  const validateAndAddFiles = (selectedFiles) => {
    const addedFiles = Array.from(selectedFiles);
    const allowedExtensions = ['pdf', 'docx', 'doc'];
    const toAdd = [];
    let hasInvalid = false;

    for (const f of addedFiles) {
      const ext = f.name.split('.').pop().toLowerCase();
      if (allowedExtensions.includes(ext)) {
        toAdd.push(f);
      } else {
        hasInvalid = true;
      }
    }

    if (toAdd.length > 0) {
      setFiles(prev => {
        // Filter out duplicate files by filename
        const filtered = toAdd.filter(newF => !prev.some(oldF => oldF.name === newF.name));
        return [...prev, ...filtered];
      });
      setStatus(null);
      setMessage('');
    }

    if (hasInvalid) {
      setStatus('error');
      setMessage('Some files were skipped. Only PDF, DOC, and DOCX formats are supported.');
    }
  };

  const removeFile = (fileName) => {
    setFiles(prev => prev.filter(f => f.name !== fileName));
    if (files.length <= 1) {
      setStatus(null);
      setMessage('');
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (files.length === 0) return;

    const targetCollection = useExisting ? selectedCollection : newCollectionName;
    if (!targetCollection || !targetCollection.trim()) {
      setStatus('error');
      setMessage('Please select or specify a target collection name.');
      return;
    }

    const finalCollectionName = targetCollection.trim().replace(/[^a-zA-Z0-9_-]/g, '');
    if (!finalCollectionName) {
      setStatus('error');
      setMessage('Invalid collection name. Use letters, numbers, hyphens, and underscores only.');
      return;
    }

    setLoading(true);
    setStatus('loading');
    setTotalFilesCount(files.length);
    setProgressPercent(0);

    let successCount = 0;
    let failedFiles = [];
    let totalChunksCreated = 0;

    for (let i = 0; i < files.length; i++) {
      const currentFile = files[i];
      setCurrentFileIndex(i + 1);
      setProgressPercent(0);
      setMessage(`Ingesting document ${i + 1} of ${files.length}: "${currentFile.name}"...`);

      const formData = new FormData();
      formData.append('file', currentFile);
      formData.append('collection_name', finalCollectionName);
      if (selectedEmbeddingId) {
        formData.append('embedding_id', selectedEmbeddingId);
      }

      try {
        const res = await fetch(`${apiUrl}/api/collections/upload`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          let detail = 'Error';
          try {
            const errData = await res.json();
            detail = errData.detail || detail;
          } catch (err) {}
          failedFiles.push(`${currentFile.name} (${detail})`);
          continue;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fileSuccess = false;
        let fileError = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // keep last incomplete line in buffer

          for (const line of lines) {
            const cleanLine = line.trim();
            if (!cleanLine) continue;

            try {
              const event = JSON.parse(cleanLine);
              if (event.type === 'progress') {
                setMessage(`[${currentFile.name}] ${event.message}`);
                if (event.percent !== undefined) {
                  setProgressPercent(event.percent);
                }
              } else if (event.status === 'success') {
                fileSuccess = true;
                totalChunksCreated += (event.chunks_created || event.chunks_upserted || 0);
                setProgressPercent(100);
              } else if (event.status === 'failed') {
                fileError = event.error || 'Ingestion failed';
                setProgressPercent(0);
              } else if (event.status === 'skipped') {
                fileError = `Skipped: ${event.error || 'Ingestion skipped'}`;
                setProgressPercent(0);
              }
            } catch (err) {
              console.error('Failed to parse NDJSON line:', cleanLine, err);
            }
          }
        }

        // Process leftover buffer
        if (buffer.trim()) {
          const cleanLine = buffer.trim();
          try {
            const event = JSON.parse(cleanLine);
            if (event.type === 'progress') {
              setMessage(`[${currentFile.name}] ${event.message}`);
              if (event.percent !== undefined) {
                setProgressPercent(event.percent);
              }
            } else if (event.status === 'success') {
              fileSuccess = true;
              totalChunksCreated += (event.chunks_created || event.chunks_upserted || 0);
              setProgressPercent(100);
            } else if (event.status === 'failed') {
              fileError = event.error || 'Ingestion failed';
              setProgressPercent(0);
            } else if (event.status === 'skipped') {
              fileError = `Skipped: ${event.error || 'Ingestion skipped'}`;
              setProgressPercent(0);
            }
          } catch (err) {
            console.error('Failed to parse leftover NDJSON line:', cleanLine, err);
          }
        }

        if (fileSuccess) {
          successCount++;
        } else {
          failedFiles.push(`${currentFile.name} (${fileError || 'Unknown Error'})`);
        }
      } catch (err) {
        console.error(err);
        failedFiles.push(`${currentFile.name} (Network or Server Error)`);
      }
    }

    if (successCount === files.length) {
      setStatus('success');
      setMessage(`Successfully ingested all ${files.length} documents! Created ${totalChunksCreated} chunks total.`);
      setFiles([]);
      if (onUploadSuccess) {
        onUploadSuccess();
      }
    } else if (successCount > 0) {
      setStatus('warning');
      setMessage(`Ingested ${successCount}/${files.length} files. Chunks created: ${totalChunksCreated}.\nFailed files: ${failedFiles.join(', ')}`);
      setFiles(prev => prev.filter(f => failedFiles.some(failed => failed.startsWith(f.name))));
      if (onUploadSuccess) {
        onUploadSuccess();
      }
    } else {
      setStatus('error');
      setMessage(`Failed to ingest documents. Errors: ${failedFiles.join(', ')}`);
    }

    setLoading(false);
  };

  return (
    <div className="upload-panel">
      <h2 className="panel-title">Ingest Documents</h2>
      <p className="panel-description">Upload PDF or Word files to parse, chunk (hierarchical), and index them into Qdrant vectors.</p>
      
      <form onSubmit={handleUpload} className="upload-form">
        <div className="collection-type-selector">
          <label className="checkbox-label">
            <input 
              type="checkbox" 
              checked={useExisting} 
              disabled={loading || !collections || collections.length === 0}
              onChange={(e) => setUseExisting(e.target.checked)}
            />
            <span>Use existing collection</span>
          </label>
        </div>

        {useExisting && collections && collections.length > 0 ? (
          <div className="input-group">
            <label htmlFor="collection-select">Select Existing Collection</label>
            <select
              id="collection-select"
              value={selectedCollection}
              onChange={(e) => setSelectedCollection(e.target.value)}
              disabled={loading}
              className="upload-select"
            >
              {collections.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        ) : (
          <div className="input-group">
            <label htmlFor="collection-input">New Collection Name</label>
            <input
              id="collection-input"
              type="text"
              placeholder="e.g. chemistry, research_papers"
              value={newCollectionName}
              onChange={(e) => setNewCollectionName(e.target.value.replace(/[^a-zA-Z0-9_-]/g, ''))}
              disabled={loading}
            />
          </div>
        )}

        <div 
          className={`dropzone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={triggerFileInput}
        >
          <input 
            type="file" 
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf,.docx,.doc"
            className="hidden-file-input"
            disabled={loading}
            multiple
          />

          <div className="dropzone-prompt">
            <span className="upload-icon">📤</span>
            <p>Drag & drop documents here, or <span className="browse-link">browse</span></p>
            <span className="file-limits">Supports PDF, DOC, DOCX (Max 20MB per file)</span>
          </div>
        </div>

        {files.length > 0 && (
          <div className="selected-files-container">
            {files.map((f) => (
              <div key={f.name} className="selected-file-row">
                <div className="selected-file-details">
                  <span className="selected-file-icon">📄</span>
                  <span className="selected-file-name-txt" title={f.name}>{f.name}</span>
                  <span className="selected-file-size-txt">{(f.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
                <button
                  type="button"
                  className="btn-remove-file-row"
                  onClick={(e) => { e.stopPropagation(); removeFile(f.name); }}
                  disabled={loading}
                  title="Remove file"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <button 
          type="submit" 
          className="btn-primary btn-upload"
          disabled={files.length === 0 || loading}
        >
          {loading ? 'Processing...' : `Upload & Ingest (${files.length} ${files.length === 1 ? 'file' : 'files'})`}
        </button>
      </form>

      {status && (
        <div className={`status-alert ${status} fade-in`}>
          <div className="status-alert-content-wrapper">
            {status === 'loading' && <span className="spinner-mini"></span>}
            <p className="status-message">{message}</p>
          </div>
          {status === 'loading' && (
            <div className="progress-container">
              <div className="progress-bar">
                <div 
                  className="progress-bar-fill" 
                  style={{ width: `${progressPercent}%` }}
                ></div>
              </div>
              <div className="progress-details">
                <span>File {currentFileIndex} of {totalFilesCount}</span>
                <span>{progressPercent}%</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
