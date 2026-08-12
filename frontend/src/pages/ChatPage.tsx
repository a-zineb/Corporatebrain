import { useEffect, useRef, useState } from 'react';
import { api } from '../api/corporateBrain';
import type { ChatResponse, DocumentItem, Source } from '../types';
import {
  ChatMessage,
  ChatTypingIndicator,
  getGroupPosition,
  shouldShowAvatar,
  type ChatMessageData,
} from '../components/chat/ChatMessage';
import { AskInputBar } from '../components/chat/AskInputBar';
import { SourcePanel } from '../components/chat/SourcePanel';
import { WelcomeCards } from '../components/chat/WelcomeCards';
import { ResizableFilesPanel } from '../components/documents/ResizableFilesPanel';
import { useToast } from '../hooks/useToast';
import { ErrorBanner } from '../components/ui/ErrorBanner';

export type Message = ChatMessageData;
type Mode = 'direct' | 'catalog' | 'ai';
interface SavedChat {
  id: string;
  title: string;
  date: string;
  document?: string;
  messages: Message[];
}

function saveHistory(messages: Message[], document?: DocumentItem) {
  if (!messages.length) return;
  const existing = JSON.parse(
    sessionStorage.getItem('cb-history') ?? '[]',
  ) as SavedChat[];
  existing.unshift({
    id: crypto.randomUUID(),
    title:
      messages.find((m) => m.role === 'user')?.text.slice(0, 64) ??
      'Conversation',
    date: new Date().toISOString(),
    document: document?.name,
    messages,
  });
  sessionStorage.setItem('cb-history', JSON.stringify(existing.slice(0, 30)));
  window.dispatchEvent(new CustomEvent('cb:history-updated'));
}

export function ChatPage() {
  const { showError, showSuccess } = useToast();
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [doc, setDoc] = useState('');
  const [mode, setMode] = useState<Mode>('direct');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState('');
  const [bannerError, setBannerError] = useState('');
  const [source, setSource] = useState<Source | null>(null);
  const [conversation, setConversation] = useState<string>();
  const conversationEnd = useRef<HTMLDivElement>(null);
  const workspaceRef = useRef<HTMLDivElement>(null);
  const uploadPicker = useRef<HTMLInputElement>(null);
  const activeDocument = docs.find((item) => item.id === doc);

  async function loadDocuments(selectFirst = false) {
    setDocsLoading(true);
    setBannerError('');
    try {
      const loaded = await api.documents();
      setDocs(loaded);
      if (selectFirst && !doc && loaded[0]) setDoc(loaded[0].id);
      return loaded;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Could not load documents.';
      setBannerError(msg);
      showError(msg, () => void loadDocuments(selectFirst));
      return [];
    } finally {
      setDocsLoading(false);
    }
  }

  useEffect(() => {
    void loadDocuments()
      .then((loaded) => {
        const raw = sessionStorage.getItem('cb-restore-chat');
        if (raw) {
          const saved = JSON.parse(raw) as SavedChat;
          setMessages(saved.messages);
          const restored = loaded.find((item) => item.name === saved.document);
          if (restored) setDoc(restored.id);
          sessionStorage.removeItem('cb-restore-chat');
        }
      })
      .catch((e) => {
        const msg = e instanceof Error ? e.message : 'Could not load documents.';
        setBannerError(msg);
        showError(msg, () => void loadDocuments());
      });
  }, []);

  useEffect(() => {
    conversationEnd.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'nearest',
    });
  }, [messages, busy]);

  function newChat() {
    saveHistory(messages, activeDocument);
    setMessages([]);
    setConversation(undefined);
    setBannerError('');
    setNotice('');
    setInput('');
  }

  function startDocumentChat(id: string) {
    saveHistory(messages, activeDocument);
    setDoc(id);
    setMessages([]);
    setConversation(undefined);
    setBannerError('');
    setNotice('');
    setMode('direct');
    setInput('');
  }

  useEffect(() => {
    const handler = () => newChat();
    window.addEventListener('cb:new-chat', handler);
    return () => window.removeEventListener('cb:new-chat', handler);
  }, [messages, activeDocument]);

  async function send(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;

    if (mode === 'catalog') {
      setMessages((m) => [
        ...m,
        { role: 'user', text: trimmed },
        {
          role: 'assistant',
          text: `Knowledge Catalog contains ${docs.length} prepared documents. Select a file on the right to start chatting.`,
        },
      ]);
      setInput('');
      return;
    }

    setMessages((m) => [...m, { role: 'user', text: trimmed }]);
    setInput('');
    setBusy(true);

    try {
      const selectedHash = doc || undefined;
      const response = await api.chat(
        trimmed,
        selectedHash,
        mode === 'ai' ? 'ai' : 'direct',
        conversation,
      );
      setConversation(response.conversation_id);
      setMessages((m) => [
        ...m,
        { role: 'assistant', text: response.answer, response },
      ]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Chat request failed.';
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          text: msg,
          error: true,
          retryText: trimmed,
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function attach(file: File) {
    setUploading(true);
    setNotice(`Preparing ${file.name}…`);
    setBannerError('');
    try {
      const uploaded = await api.upload(file);
      const loaded = await loadDocuments(false);
      setDocs(loaded);
      saveHistory(messages, activeDocument);
      setMessages([]);
      setConversation(undefined);
      setDoc(uploaded.id);
      setMode('direct');
      setNotice(`${uploaded.name} is ready. A new chat has started.`);
      showSuccess('Document uploaded', `${uploaded.name} is ready to chat.`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Upload failed.';
      setBannerError(msg);
      showError(msg, () => void attach(file));
      setNotice('');
    } finally {
      setUploading(false);
    }
  }

  if (source) {
    return <SourcePanel source={source} onClose={() => setSource(null)} />;
  }

  const typingLabel =
    mode === 'ai'
      ? 'Corporate Brain is generating…'
      : 'Corporate Brain is searching…';

  return (
    <div className="chat-workspace" ref={workspaceRef}>
      <section className="chat-main glass-card">
        <div className="chat-main__body">
          <div className="conversation">
            {messages.length === 0 && (
              <div className="welcome">
                <h1>
                  Welcome to Corporate Brain, what should we start with today?
                </h1>
                <p>
                  Choose a mode below or pick a file on the right to begin.
                </p>
                <WelcomeCards
                  onSelectMode={setMode}
                  onUpload={() => uploadPicker.current?.click()}
                />
                <input
                  ref={uploadPicker}
                  hidden
                  type="file"
                  accept=".pdf,.docx,.doc,.xlsx,.csv,.zip"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void attach(file);
                    e.currentTarget.value = '';
                  }}
                />
                {mode !== 'direct' && (
                  <p className="welcome__mode">
                    Active mode:{' '}
                    <strong>
                      {mode === 'ai' ? 'AI Answer' : 'Catalog'}
                    </strong>
                  </p>
                )}
              </div>
            )}

            {messages.map((message, index) => (
              <ChatMessage
                key={`${index}-${message.role}-${message.text.slice(0, 24)}`}
                message={message}
                showAvatar={shouldShowAvatar(messages, index)}
                groupPosition={getGroupPosition(messages, index)}
                onOpenSource={setSource}
                onRetry={(text) => void send(text)}
              />
            ))}

            {busy && <ChatTypingIndicator label={typingLabel} />}

            {notice && <div className="notice">{notice}</div>}
            {bannerError && (
              <ErrorBanner
                message={bannerError}
                onRetry={() => void loadDocuments()}
                onDismiss={() => setBannerError('')}
              />
            )}
            <div ref={conversationEnd} />
          </div>
        </div>

        <div className="chat-composer-area">
          <AskInputBar
            value={input}
            onChange={setInput}
            onSubmit={() => void send(input)}
            onAttach={attach}
            placeholder="Ask Anything…"
            busy={busy}
            disabled={uploading}
          />
          {uploading && (
            <p className="composer__status">Preparing document…</p>
          )}
        </div>
      </section>

      <ResizableFilesPanel
        documents={docs}
        active={doc}
        onSelect={startDocumentChat}
        onUpload={attach}
        containerRef={workspaceRef}
        loading={docsLoading}
      />
    </div>
  );
}
