import React, { useState, useRef, useEffect } from 'react';
import './ChatWindow.css';

// Simple Markdown rendering helper for structured, academic text formatting
const renderMarkdown = (text) => {
  if (!text) return '';
  
  const lines = text.split('\n');
  const elements = [];
  let currentList = [];
  let listType = null; // 'ul' or 'ol'
  
  const flushList = (key) => {
    if (currentList.length > 0) {
      if (listType === 'ul') {
        elements.push(<ul key={`ul-${key}`} className="chat-markdown-ul">{...currentList}</ul>);
      } else {
        elements.push(<ol key={`ol-${key}`} className="chat-markdown-ol">{...currentList}</ol>);
      }
      currentList = [];
      listType = null;
    }
  };

  const parseInline = (lineText) => {
    // Replace **text** with <strong>text</strong>
    const parts = lineText.split('**');
    const nodes = [];
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 1) {
        nodes.push(<strong key={`bold-${i}`}>{parts[i]}</strong>);
      } else {
        // Also handle inline code `code`
        const codeParts = parts[i].split('`');
        for (let j = 0; j < codeParts.length; j++) {
          if (j % 2 === 1) {
            nodes.push(<code key={`code-${i}-${j}`}>{codeParts[j]}</code>);
          } else {
            nodes.push(codeParts[j]);
          }
        }
      }
    }
    return nodes;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    
    // Empty lines
    if (trimmed === '') {
      flushList(i);
      continue;
    }

    // Headings
    if (trimmed.startsWith('### ')) {
      flushList(i);
      elements.push(<h3 key={`h3-${i}`} className="chat-markdown-h3">{parseInline(trimmed.substring(4))}</h3>);
      continue;
    }
    if (trimmed.startsWith('## ')) {
      flushList(i);
      elements.push(<h2 key={`h2-${i}`} className="chat-markdown-h2">{parseInline(trimmed.substring(3))}</h2>);
      continue;
    }
    if (trimmed.startsWith('# ')) {
      flushList(i);
      elements.push(<h1 key={`h1-${i}`} className="chat-markdown-h1">{parseInline(trimmed.substring(2))}</h1>);
      continue;
    }

    // Bullet lists (support '-', '*', and '•')
    const isBullet = trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ');
    // Numbered lists
    const isNumbered = /^\d+\.\s/.test(trimmed);

    if (isBullet) {
      if (listType !== 'ul') {
        flushList(i);
        listType = 'ul';
      }
      const itemText = trimmed.substring(2);
      currentList.push(<li key={`li-${i}`}>{parseInline(itemText)}</li>);
    } else if (isNumbered) {
      if (listType !== 'ol') {
        flushList(i);
        listType = 'ol';
      }
      const match = trimmed.match(/^\d+\.\s/);
      const itemText = trimmed.substring(match[0].length);
      currentList.push(<li key={`li-${i}`}>{parseInline(itemText)}</li>);
    } else {
      flushList(i);
      elements.push(<p key={`p-${i}`} className="chat-markdown-p">{parseInline(line)}</p>);
    }
  }
  flushList(lines.length);

  return elements;
};

export default function ChatWindow({ 
  apiUrl, 
  activeCollection, 
  currentSessionId, 
  selectedLlmId, 
  selectedEmbeddingId, 
  onMessageSent 
}) {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [currentQueryUpdate, setCurrentQueryUpdate] = useState(null);
  
  // Feedback states
  const [feedbackMessageId, setFeedbackMessageId] = useState(null);
  const [feedbackRating, setFeedbackRating] = useState(null); // 'up' or 'down'
  const [feedbackComment, setFeedbackComment] = useState('');
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Fetch session history when currentSessionId changes
  useEffect(() => {
    if (!currentSessionId) {
      setMessages([]);
      return;
    }

    const fetchSessionHistory = async () => {
      try {
        const res = await fetch(`${apiUrl}/api/sessions/${currentSessionId}/history`);
        if (res.ok) {
          const data = await res.json();
          setMessages(data);
        } else {
          console.error("Failed to fetch session history");
        }
      } catch (err) {
        console.error("Error fetching session history", err);
      }
    };

    fetchSessionHistory();
  }, [currentSessionId, apiUrl]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentQueryUpdate]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || streaming) return;

    const userPrompt = inputValue.trim();
    setInputValue('');
    setStreaming(true);
    setCurrentQueryUpdate(null);

    // Create a unique message ID for the assistant response
    const assistantMsgId = `assistant-msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Add user message to state
    const newUserMessage = { role: 'user', content: userPrompt, id: `user-msg-${Date.now()}` };
    setMessages(prev => [...prev, newUserMessage]);

    // Prepare assistant message template
    const newAssistantMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      citations: [],
      feedbackSubmitted: null, // 'up' or 'down'
      originalQuery: userPrompt
    };

    setMessages(prev => [...prev, newAssistantMessage]);

    // Fetch SSE stream
    try {
      // Map message history to standard format (role/content)
      const chatHistory = messages.map(m => ({
        role: m.role,
        content: m.content
      }));

      const response = await fetch(`${apiUrl}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userPrompt,
          history: chatHistory,
          collection_name: activeCollection,
          session_id: currentSessionId,
          llm_id: selectedLlmId,
          embedding_id: selectedEmbeddingId,
          user_msg_id: newUserMessage.id,
          assistant_msg_id: assistantMsgId
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        
        // Save the incomplete line back to buffer
        buffer = lines.pop();

        for (const line of lines) {
          const cleanLine = line.trim();
          if (!cleanLine.startsWith('data:')) continue;

          try {
            const jsonStr = cleanLine.substring(5).trim();
            const data = JSON.parse(jsonStr);

            if (data.type === 'token') {
              setMessages(prev => prev.map(msg => {
                if (msg.id === assistantMsgId) {
                  return { ...msg, content: msg.content + data.content };
                }
                return msg;
              }));
            } else if (data.type === 'citations') {
              setMessages(prev => prev.map(msg => {
                if (msg.id === assistantMsgId) {
                  return { ...msg, citations: data.chunks || [] };
                }
                return msg;
              }));
            } else if (data.type === 'query_update') {
              setCurrentQueryUpdate(data.query);
            } else if (data.type === 'error') {
              setMessages(prev => prev.map(msg => {
                if (msg.id === assistantMsgId) {
                  return { ...msg, content: msg.content + `\n\n⚠️ Error: ${data.message}` };
                }
                return msg;
              }));
            }
          } catch (e) {
            console.error("Failed to parse SSE line", e);
          }
        }
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMsgId) {
          return { ...msg, content: msg.content + `\n\n⚠️ Connection Error: Could not stream response. Check backend connection.` };
        }
        return msg;
      }));
    } finally {
      setStreaming(false);
      setCurrentQueryUpdate(null);
      if (onMessageSent) {
        onMessageSent();
      }
    }
  };

  const handleOpenFeedback = (msgId, rating) => {
    setFeedbackMessageId(msgId);
    setFeedbackRating(rating);
    setFeedbackComment('');
  };

  const handleSubmitFeedback = async () => {
    if (!feedbackMessageId || !feedbackRating) return;

    setSubmittingFeedback(true);
    const targetMsg = messages.find(m => m.id === feedbackMessageId);
    
    const payload = {
      message_id: feedbackMessageId,
      session_id: currentSessionId || 'default-session-id',
      prompt: targetMsg?.originalQuery || '',
      answer: targetMsg?.content || '',
      feedback: feedbackRating,
      comment: feedbackComment.trim() || null,
      retrieved_chunks: targetMsg?.citations.map(c => ({
        source: c.source,
        text: c.text,
        page_number: c.page_number,
        score: c.score
      })) || [],
      timestamp: new Date().toISOString()
    };

    try {
      const res = await fetch(`${apiUrl}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        // Mark feedback as submitted in state
        setMessages(prev => prev.map(msg => {
          if (msg.id === feedbackMessageId) {
            return { ...msg, feedbackSubmitted: feedbackRating };
          }
          return msg;
        }));
        // Reset feedback modal/dialog
        setFeedbackMessageId(null);
        setFeedbackRating(null);
      } else {
        alert("Failed to submit feedback.");
      }
    } catch (err) {
      console.error("Feedback error", err);
      alert("Error submitting feedback.");
    } finally {
      setSubmittingFeedback(false);
    }
  };

  return (
    <div className="chat-window">
      <div className="chat-header">
        <div className="header-info">
          <h2>Agent Session</h2>
          <span className="scope-pill">
            Scope: {activeCollection ? `📁 ${activeCollection}` : '🔍 Global Search'}
          </span>
        </div>
        {streaming && <div className="activity-indicator">Agent Active</div>}
      </div>

      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-screen fade-in">
            <span className="welcome-logo">🎓</span>
            <h3>Welcome to Eva Research RAG</h3>
            <p>Select a collection on the left sidebar, upload documents, or start asking questions about indexed articles.</p>
            <div className="query-examples">
              <button onClick={() => setInputValue("What are the key conclusions in the papers?")}>
                💡 What are the key conclusions in the papers?
              </button>
              <button onClick={() => setInputValue("Explain the experimental methodologies used.")}>
                💡 Explain the experimental methodologies used.
              </button>
            </div>
          </div>
        )}

        {messages.map((msg) => {
          const isUser = msg.role === 'user';
          return (
            <div key={msg.id} className={`message-row ${isUser ? 'user' : 'assistant'} fade-in`}>
              <div className="message-avatar">
                {isUser ? '👤' : '🤖'}
              </div>
              <div className="message-bubble-wrapper">
                <div className="message-bubble">
                  {msg.content ? (
                    renderMarkdown(msg.content)
                  ) : (
                    streaming && msg.id === messages[messages.length - 1].id ? (
                      <span className="loading-dots"><span>.</span><span>.</span><span>.</span></span>
                    ) : ''
                  )}
                </div>

                {/* Query Rewrite Indicator */}
                {!isUser && streaming && currentQueryUpdate && msg.id === messages[messages.length - 1].id && (
                  <div className="query-rewrite-status pulse-glow">
                    🔄 Corrective RAG: Optimizing search query to <strong>"{currentQueryUpdate}"</strong>
                  </div>
                )}

                {/* Citations panel */}
                {!isUser && msg.citations && msg.citations.length > 0 && (
                  <div className="citations-block">
                    <span className="citations-header">📚 Retrieved Citations:</span>
                    <div className="citations-accordions">
                      {msg.citations.map((cite, index) => (
                        <details key={index} className="citation-accordion card-glass">
                          <summary className="citation-summary">
                            <span className="cite-arrow">▶</span>
                            <span className="cite-source">{cite.file_name || cite.source}</span>
                            <span className="cite-page-badge">Page {cite.page_number}</span>
                            <span className="cite-score-badge">{(cite.score * 100).toFixed(0)}% match</span>
                          </summary>
                          <div className="citation-details-content fade-in">
                            {cite.doc_type === "image" && cite.asset_path && (
                              <div className="citation-image-container">
                                <img 
                                  src={`${apiUrl}/api/assets/${cite.asset_path.split(/[\\/]/).pop()}`} 
                                  alt={cite.text || "Extracted illustration"} 
                                  className="citation-image"
                                />
                              </div>
                            )}
                            <p className="citation-text">
                              {cite.doc_type === "image" ? <strong>Visual Caption: </strong> : ""}
                              "{cite.text}"
                            </p>
                          </div>
                        </details>
                      ))}
                    </div>
                  </div>
                )}

                {/* Feedback Panel */}
                {!isUser && msg.content && (
                  <div className="message-feedback-actions">
                    <button 
                      className={`btn-feedback up ${msg.feedbackSubmitted === 'up' ? 'active' : ''}`}
                      onClick={() => handleOpenFeedback(msg.id, 'up')}
                      disabled={msg.feedbackSubmitted !== null}
                      title="Helpful Answer"
                    >
                      👍 {msg.feedbackSubmitted === 'up' && 'Thanks!'}
                    </button>
                    <button 
                      className={`btn-feedback down ${msg.feedbackSubmitted === 'down' ? 'active' : ''}`}
                      onClick={() => handleOpenFeedback(msg.id, 'down')}
                      disabled={msg.feedbackSubmitted !== null}
                      title="Unhelpful / Irrelevant"
                    >
                      👎 {msg.feedbackSubmitted === 'down' && 'Flagged'}
                    </button>
                  </div>
                )}

                {/* Inline Comment Dialog for Feedback */}
                {feedbackMessageId === msg.id && (
                  <div className="feedback-dialog-card fade-in">
                    <h4>Add optional comments to improve the RAG query loops:</h4>
                    <textarea 
                      value={feedbackComment}
                      onChange={(e) => setFeedbackComment(e.target.value)}
                      placeholder={feedbackRating === 'up' ? 'What made this answer good?' : 'Why was this answer irrelevant or wrong?'}
                      rows={3}
                    />
                    <div className="dialog-buttons">
                      <button 
                        className="btn-cancel"
                        onClick={() => { setFeedbackMessageId(null); setFeedbackRating(null); }}
                      >
                        Cancel
                      </button>
                      <button 
                        className="btn-submit"
                        onClick={handleSubmitFeedback}
                        disabled={submittingFeedback}
                      >
                        {submittingFeedback ? 'Submitting...' : 'Submit Feedback'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-footer">
        <form onSubmit={handleSend} className="input-form">
          <input
            type="text"
            placeholder={streaming ? "Agent is writing answer..." : "Type research query... (citations will display below response)"}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            disabled={streaming}
          />
          <button type="submit" className="btn-send" disabled={!inputValue.trim() || streaming}>
            {streaming ? '⏳' : '➔'}
          </button>
        </form>
      </div>
    </div>
  );
}
