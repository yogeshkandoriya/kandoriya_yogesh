/**
 * Summary AI — Next-Generation Frontend Controller
 * Complete responsive document workspace with grounded RAG,
 * SQLite persistence, document & chat history, citation inspection, and multi-format exporter.
 */

// -------------------------------------------------------------------------- //
// Configuration & Environment
// -------------------------------------------------------------------------- //
const API_URL = (location.protocol === 'http:' || location.protocol === 'https:')
    ? '' : 'http://localhost:8000';

// App State
const state = {
    fileId: null,
    docData: null,
    isBusy: false,
    abortController: null,
    pollTimer: null,
    activeTab: 'chatTab',
    currentPage: 1,
    totalPages: 1,
    keywords: [],
    sourcesCache: {},
    speechSynthUtterance: null,
    cachedHistory: [],
};

// DOM References Cache
const dom = {
    // Theme & Navigation
    themeToggleBtn: document.getElementById('themeToggleBtn'),
    themeIcon: document.getElementById('themeIcon'),
    brandHome: document.getElementById('brandHome'),
    navNewDocBtn: document.getElementById('navNewDocBtn'),
    modelStatusPill: document.getElementById('modelStatusPill'),
    modelNameText: document.getElementById('modelNameText'),
    serverDot: document.getElementById('serverDot'),
    mobileMenuBtn: document.getElementById('mobileMenuBtn'),
    headerDocInfo: document.getElementById('headerDocInfo'),
    headerDocName: document.getElementById('headerDocName'),

    // History Navigation & Landing Shelf
    historyNavBtn: document.getElementById('historyNavBtn'),
    historyNavBadge: document.getElementById('historyNavBadge'),
    landingRecentSection: document.getElementById('landingRecentSection'),
    recentDocsLandingGrid: document.getElementById('recentDocsLandingGrid'),
    recentDocsCount: document.getElementById('recentDocsCount'),
    clearAllHistoryLandingBtn: document.getElementById('clearAllHistoryLandingBtn'),

    // Views
    initialView: document.getElementById('initialView'),
    activeWorkspace: document.getElementById('activeWorkspace'),
    workspaceSidebar: document.getElementById('workspaceSidebar'),
    sidebarBackdrop: document.getElementById('sidebarBackdrop'),

    // Upload Elements
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    uploadContent: document.getElementById('uploadContent'),
    uploadProgress: document.getElementById('uploadProgress'),
    progressStageTitle: document.getElementById('progressStageTitle'),
    progressStageDetail: document.getElementById('progressStageDetail'),
    stepUpload: document.getElementById('stepUpload'),
    stepExtract: document.getElementById('stepExtract'),
    stepIndex: document.getElementById('stepIndex'),
    stepReady: document.getElementById('stepReady'),

    // Sidebar Data & Switcher
    docTitle: document.getElementById('docTitle'),
    statPages: document.getElementById('statPages'),
    statWords: document.getElementById('statWords'),
    statChunks: document.getElementById('statChunks'),
    keywordsList: document.getElementById('keywordsList'),
    kwCountBadge: document.getElementById('kwCountBadge'),
    sidebarRecentSection: document.getElementById('sidebarRecentSection'),
    sidebarRecentList: document.getElementById('sidebarRecentList'),
    sidebarManageHistoryBtn: document.getElementById('sidebarManageHistoryBtn'),
    closeDocBtn: document.getElementById('closeDocBtn'),
    exportDropdownBtn: document.getElementById('exportDropdownBtn'),
    exportMenu: document.getElementById('exportMenu'),

    // Tabs
    navTabs: document.querySelectorAll('.nav-tab'),
    tabPanes: document.querySelectorAll('.tab-pane'),
    clearChatBtn: document.getElementById('clearChatBtn'),

    // Chat Tab
    chatHistory: document.getElementById('chatHistory'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    stopGenBtn: document.getElementById('stopGenBtn'),
    suggestionChips: document.getElementById('suggestionChips'),
    slashHelper: document.getElementById('slashHelper'),

    // Document Reader Tab
    prevPageBtn: document.getElementById('prevPageBtn'),
    nextPageBtn: document.getElementById('nextPageBtn'),
    readerPageInput: document.getElementById('readerPageInput'),
    readerTotalPages: document.getElementById('readerTotalPages'),
    readerSearchInput: document.getElementById('readerSearchInput'),
    clearSearchBtn: document.getElementById('clearSearchBtn'),
    pageCardBadge: document.getElementById('pageCardBadge'),
    pageWordCount: document.getElementById('pageWordCount'),
    pageTextContent: document.getElementById('pageTextContent'),
    askAboutPageBtn: document.getElementById('askAboutPageBtn'),
    viewRawPdfBtn: document.getElementById('viewRawPdfBtn'),

    // Topic Explorer Tab
    topicsExplorerGrid: document.getElementById('topicsExplorerGrid'),

    // History Modal
    historyModal: document.getElementById('historyModal'),
    historyModalTitle: document.getElementById('historyModalTitle'),
    historyModalCountBadge: document.getElementById('historyModalCountBadge'),
    historyModalList: document.getElementById('historyModalList'),
    historySearchInput: document.getElementById('historySearchInput'),
    closeHistoryModal: document.getElementById('closeHistoryModal'),
    clearAllHistoryModalBtn: document.getElementById('clearAllHistoryModalBtn'),
    historyCloseBtn: document.getElementById('historyCloseBtn'),

    // Citation Passage Modal
    passageModal: document.getElementById('passageModal'),
    modalPageTitle: document.getElementById('modalPageTitle'),
    modalScoreBadge: document.getElementById('modalScoreBadge'),
    modalExcerptText: document.getElementById('modalExcerptText'),
    closePassageModal: document.getElementById('closePassageModal'),
    modalJumpReaderBtn: document.getElementById('modalJumpReaderBtn'),
    modalAskPassageBtn: document.getElementById('modalAskPassageBtn'),

    // Diagnostics Modal
    diagnosticsModal: document.getElementById('diagnosticsModal'),
    diagPrimaryModel: document.getElementById('diagPrimaryModel'),
    diagFallbacks: document.getElementById('diagFallbacks'),
    diagKeyStatus: document.getElementById('diagKeyStatus'),
    closeDiagnosticsModal: document.getElementById('closeDiagnosticsModal'),
    diagOkBtn: document.getElementById('diagOkBtn'),

    // Toast
    toast: document.getElementById('toast'),
    toastIcon: document.getElementById('toastIcon'),
    toastMessage: document.getElementById('toastMessage'),
    toastCloseBtn: document.getElementById('toastCloseBtn'),
};

// -------------------------------------------------------------------------- //
// Initialization
// -------------------------------------------------------------------------- //
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initEventListeners();
    checkHealth();
    loadHistoryList();
});

function initTheme() {
    const savedTheme = localStorage.getItem('summary_ai_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function updateThemeIcon(theme) {
    if (dom.themeIcon) {
        dom.themeIcon.className = theme === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun';
    }
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('summary_ai_theme', next);
    updateThemeIcon(next);
}

// -------------------------------------------------------------------------- //
// Event Listeners
// -------------------------------------------------------------------------- //
function initEventListeners() {
    // Theme toggle
    dom.themeToggleBtn.addEventListener('click', toggleTheme);
    
    // Brand home click
    dom.brandHome.addEventListener('click', () => {
        if (state.fileId && confirm('Return to home screen? Your document & chat history remain saved.')) {
            closeDocument();
        }
    });

    // Mobile Drawer
    dom.mobileMenuBtn.addEventListener('click', toggleSidebarDrawer);
    dom.sidebarBackdrop.addEventListener('click', closeSidebarDrawer);

    // Diagnostics Modal
    dom.modelStatusPill.addEventListener('click', showDiagnosticsModal);
    dom.closeDiagnosticsModal.addEventListener('click', hideDiagnosticsModal);
    dom.diagOkBtn.addEventListener('click', hideDiagnosticsModal);

    // History Navigation & Modals
    dom.historyNavBtn.addEventListener('click', showHistoryModal);
    dom.sidebarManageHistoryBtn.addEventListener('click', showHistoryModal);
    dom.closeHistoryModal.addEventListener('click', hideHistoryModal);
    dom.historyCloseBtn.addEventListener('click', hideHistoryModal);
    dom.clearAllHistoryLandingBtn.addEventListener('click', handleClearAllHistory);
    dom.clearAllHistoryModalBtn.addEventListener('click', handleClearAllHistory);
    dom.historySearchInput.addEventListener('input', handleHistorySearch);

    // Upload Actions
    dom.uploadZone.addEventListener('click', (e) => {
        if (e.target.closest('.upload-processing')) return;
        dom.fileInput.click();
    });
    dom.uploadZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') dom.fileInput.click();
    });
    dom.fileInput.addEventListener('change', () => {
        if (dom.fileInput.files.length) handleFileUpload(dom.fileInput.files[0]);
        dom.fileInput.value = '';
    });

    // Drag & Drop
    ['dragover', 'dragenter'].forEach(ev => {
        dom.uploadZone.addEventListener(ev, (e) => {
            e.preventDefault();
            dom.uploadZone.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach(ev => {
        dom.uploadZone.addEventListener(ev, () => {
            dom.uploadZone.classList.remove('dragover');
        });
    });
    dom.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
    });

    // Navigation Tabs
    dom.navTabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Summary Mode Buttons
    document.querySelectorAll('.summary-action-btn').forEach(btn => {
        btn.addEventListener('click', () => triggerSummary(btn.dataset.length, btn));
    });

    // Chat Composer
    dom.sendBtn.addEventListener('click', handleChatSubmit);
    dom.stopGenBtn.addEventListener('click', interruptGeneration);
    dom.chatInput.addEventListener('keydown', handleInputKeydown);
    dom.chatInput.addEventListener('input', handleChatInputChanges);

    // Slash Helper clicks
    document.querySelectorAll('.slash-item').forEach(item => {
        item.addEventListener('click', () => {
            const cmd = item.dataset.cmd;
            handleSlashCommand(cmd);
            dom.slashHelper.classList.add('hidden');
        });
    });

    // Document Reader Navigation
    dom.prevPageBtn.addEventListener('click', () => navigatePage(-1));
    dom.nextPageBtn.addEventListener('click', () => navigatePage(1));
    dom.readerPageInput.addEventListener('change', () => {
        const p = parseInt(dom.readerPageInput.value, 10);
        if (!isNaN(p)) setReaderPage(p);
    });
    dom.readerSearchInput.addEventListener('input', handleReaderSearch);
    dom.clearSearchBtn.addEventListener('click', () => {
        dom.readerSearchInput.value = '';
        dom.clearSearchBtn.classList.add('hidden');
        renderCurrentPage();
    });
    dom.askAboutPageBtn.addEventListener('click', () => {
        switchTab('chatTab');
        dom.chatInput.value = `Explain the main concepts covered on Page ${state.currentPage} of this document.`;
        autoResizeTextarea();
        handleChatSubmit();
    });
    dom.viewRawPdfBtn.addEventListener('click', () => {
        if (state.fileId) window.open(`${API_URL}/pdf/${state.fileId}`, '_blank');
    });

    // Clear Chat
    dom.clearChatBtn.addEventListener('click', handleClearActiveChat);

    // Export Dropdown
    dom.exportDropdownBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        dom.exportMenu.classList.toggle('hidden');
    });
    document.addEventListener('click', () => dom.exportMenu.classList.add('hidden'));

    document.querySelectorAll('.export-opt').forEach(opt => {
        opt.addEventListener('click', (e) => {
            e.stopPropagation();
            dom.exportMenu.classList.add('hidden');
            handleExport(opt.dataset.fmt);
        });
    });

    // Global Keybindings (Escape to stop/close, Alt+N for new doc)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (state.isBusy) interruptGeneration();
            hidePassageModal();
            hideDiagnosticsModal();
            hideHistoryModal();
            closeSidebarDrawer();
        } else if (e.altKey && (e.key === 'n' || e.key === 'N')) {
            e.preventDefault();
            closeDocument();
        }
    });

    // Passage Modal Controls
    dom.closePassageModal.addEventListener('click', hidePassageModal);
    dom.modalJumpReaderBtn.addEventListener('click', () => {
        const page = parseInt(dom.modalJumpReaderBtn.dataset.page || '1', 10);
        hidePassageModal();
        switchTab('readerTab');
        setReaderPage(page);
    });
    dom.modalAskPassageBtn.addEventListener('click', () => {
        const text = dom.modalExcerptText.textContent;
        const page = dom.modalJumpReaderBtn.dataset.page || '1';
        hidePassageModal();
        switchTab('chatTab');
        dom.chatInput.value = `From Page ${page}: "${text.slice(0, 150)}..." — Can you elaborate on this?`;
        autoResizeTextarea();
        handleChatSubmit();
    });

    // Header buttons
    dom.navNewDocBtn.addEventListener('click', closeDocument);
    dom.closeDocBtn.addEventListener('click', closeDocument);
    dom.toastCloseBtn.addEventListener('click', hideToast);
}

// -------------------------------------------------------------------------- //
// Health Check & Diagnostics
// -------------------------------------------------------------------------- //
async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/health`);
        const data = await res.json();
        dom.modelNameText.textContent = data.model ? data.model.split('/').pop() : 'online';
        dom.serverDot.className = 'status-indicator online';
        
        dom.diagPrimaryModel.textContent = data.model || 'nvidia/nemotron-3.5-lightning:free';
        dom.diagFallbacks.innerHTML = (data.fallbacks || []).map(f => `<div class="code-badge" style="margin-top:4px">${f}</div>`).join('') || 'None';
        dom.diagKeyStatus.textContent = data.api_key_configured ? 'Configured & Active' : 'Missing (Add OPENROUTER_API_KEY in Render → Environment)';
        dom.diagKeyStatus.style.color = data.api_key_configured ? 'var(--success)' : 'var(--danger)';
    } catch {
        dom.modelNameText.textContent = 'server offline';
        dom.serverDot.className = 'status-indicator offline';
        showToast('Cannot connect to backend server', 'error');
    }
}

function showDiagnosticsModal() {
    dom.diagnosticsModal.classList.remove('hidden');
}

function hideDiagnosticsModal() {
    dom.diagnosticsModal.classList.add('hidden');
}

// -------------------------------------------------------------------------- //
// Document & Chat History System (SQLite Persistence)
// -------------------------------------------------------------------------- //
async function loadHistoryList() {
    try {
        const res = await fetch(`${API_URL}/history/documents`);
        const data = await res.json();
        if (res.ok && data.documents) {
            state.cachedHistory = data.documents;
            renderHistoryUI(data.documents);
        }
    } catch (err) {
        console.warn('Could not load history list:', err);
    }
}

function renderHistoryUI(documents) {
    const count = documents.length;

    // Update Header Badge
    if (count > 0) {
        dom.historyNavBadge.textContent = count;
        dom.historyNavBadge.classList.remove('hidden');
    } else {
        dom.historyNavBadge.classList.add('hidden');
    }

    // Render Landing Page Recent Docs Shelf
    if (count > 0) {
        dom.landingRecentSection.classList.remove('hidden');
        dom.recentDocsCount.textContent = count;
        dom.recentDocsLandingGrid.innerHTML = documents.slice(0, 6).map(doc => `
            <div class="recent-doc-card" onclick="openDocumentFromHistory('${doc.file_id}')">
                <div class="recent-doc-top">
                    <div class="recent-doc-icon"><i class="fa-solid fa-file-pdf"></i></div>
                    <div class="recent-doc-meta">
                        <div class="recent-doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                        <div class="recent-doc-time"><i class="fa-regular fa-clock"></i> ${formatTimeAgo(doc.updated_at || doc.created_at)}</div>
                    </div>
                </div>
                <div class="recent-doc-stats">
                    <span class="recent-stat-pill"><i class="fa-solid fa-file"></i> ${doc.page_count || 1}p</span>
                    <span class="recent-stat-pill"><i class="fa-solid fa-comments"></i> ${doc.message_count || 0} msgs</span>
                    <div class="recent-doc-actions">
                        <button class="recent-del-btn" title="Delete document" onclick="deleteDocumentFromHistory('${doc.file_id}', event)">
                            <i class="fa-solid fa-trash-can"></i>
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    } else {
        dom.landingRecentSection.classList.add('hidden');
    }

    // Render Sidebar Switcher List
    if (count > 0) {
        dom.sidebarRecentSection.classList.remove('hidden');
        dom.sidebarRecentList.innerHTML = documents.slice(0, 5).map(doc => `
            <div class="sidebar-recent-item ${doc.file_id === state.fileId ? 'active' : ''}" 
                 onclick="openDocumentFromHistory('${doc.file_id}')" 
                 title="${escapeHtml(doc.filename)}">
                <i class="fa-solid fa-file-pdf" style="color:${doc.file_id === state.fileId ? 'var(--primary-light)' : '#ef4444'}"></i>
                <span class="doc-name">${escapeHtml(doc.filename)}</span>
                <span class="doc-badge">${doc.page_count || 1}p</span>
            </div>
        `).join('');
    } else {
        dom.sidebarRecentSection.classList.add('hidden');
    }

    // Render Full History Modal List
    renderHistoryModalList(documents);
}

function renderHistoryModalList(documents) {
    dom.historyModalCountBadge.textContent = `${documents.length} Document${documents.length === 1 ? '' : 's'}`;
    
    if (!documents.length) {
        dom.historyModalList.innerHTML = `
            <div class="history-empty-state">
                <i class="fa-solid fa-folder-open"></i>
                <h4>No documents in history</h4>
                <p>Upload a PDF document to start analyzing, summarizing, and saving your chat history.</p>
            </div>
        `;
        return;
    }

    dom.historyModalList.innerHTML = documents.map(doc => `
        <div class="history-modal-item" id="histItem-${doc.file_id}">
            <div class="history-item-left">
                <div class="history-item-icon"><i class="fa-solid fa-file-pdf"></i></div>
                <div class="history-item-content">
                    <div class="history-item-title" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                    <div class="history-item-meta">
                        <span><i class="fa-regular fa-clock"></i> ${formatDateString(doc.created_at)}</span>
                        <span>•</span>
                        <span><i class="fa-solid fa-file"></i> ${doc.page_count || 1} pages</span>
                        <span>•</span>
                        <span><i class="fa-solid fa-comments"></i> ${doc.message_count || 0} messages</span>
                    </div>
                </div>
            </div>
            <div class="history-item-right">
                <button class="btn btn-primary btn-sm" onclick="openDocumentFromHistory('${doc.file_id}')">
                    <i class="fa-solid fa-arrow-up-right-from-square"></i> Open
                </button>
                <button class="icon-btn-sm" title="Delete document and history" onclick="deleteDocumentFromHistory('${doc.file_id}', event)">
                    <i class="fa-solid fa-trash-can" style="color:var(--danger)"></i>
                </button>
            </div>
        </div>
    `).join('');
}

function showHistoryModal() {
    dom.historySearchInput.value = '';
    renderHistoryModalList(state.cachedHistory);
    dom.historyModal.classList.remove('hidden');
    dom.historySearchInput.focus();
}

function hideHistoryModal() {
    dom.historyModal.classList.add('hidden');
}

function handleHistorySearch() {
    const q = dom.historySearchInput.value.toLowerCase().trim();
    if (!q) {
        renderHistoryModalList(state.cachedHistory);
        return;
    }
    const filtered = state.cachedHistory.filter(d => 
        (d.filename || '').toLowerCase().includes(q)
    );
    renderHistoryModalList(filtered);
}

async function openDocumentFromHistory(fileId) {
    hideHistoryModal();
    if (fileId === state.fileId) {
        switchTab('chatTab');
        return;
    }

    setUploadState(true, 'Loading Document from History…', 'Retrieving saved index and conversation history', 'stepIndex');
    
    try {
        const res = await fetch(`${API_URL}/history/${fileId}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Could not load document history');

        const doc = data.document;
        const messages = data.messages || [];

        state.fileId = doc.file_id;
        state.totalPages = Math.max(1, doc.page_count || 1);
        state.currentPage = 1;

        // Populate Workspace UI
        dom.initialView.classList.add('hidden');
        dom.activeWorkspace.classList.remove('hidden');
        dom.navNewDocBtn.classList.remove('hidden');
        dom.headerDocInfo.classList.remove('hidden');

        dom.docTitle.textContent = doc.filename;
        dom.docTitle.title = doc.filename;
        dom.headerDocName.textContent = doc.filename;

        dom.statPages.textContent = doc.page_count ?? '1';
        dom.statWords.textContent = (doc.word_count ?? 0).toLocaleString();
        dom.statChunks.textContent = doc.chunk_count ?? '0';

        dom.readerPageInput.max = state.totalPages;
        dom.readerTotalPages.textContent = state.totalPages;

        // Reset & Restore Chat History Messages
        dom.chatHistory.innerHTML = '';
        renderWelcomeBanner();

        if (messages.length > 0) {
            messages.forEach(msg => {
                if (msg.sender === 'user') {
                    appendUserMessage(msg.content);
                } else if (msg.sender === 'ai') {
                    const aiMsgEl = appendThinkingMessage();
                    renderAiResponse(aiMsgEl, msg.content, {
                        sources: msg.sources || [],
                        footer: msg.model ? `${msg.model} · restored` : 'Restored',
                    });
                }
            });
        }

        // Fetch reader data, keywords, suggested questions
        loadDocumentReaderData();
        loadKeywords();
        loadSuggestedQuestions();

        switchTab('chatTab');
        setUploadState(false);
        showToast(`Loaded "${doc.filename}" with ${messages.length} saved messages!`, 'success');
        
        // Refresh history list to update active highlight
        loadHistoryList();
    } catch (err) {
        setUploadState(false);
        showToast(err.message, 'error');
    }
}

async function deleteDocumentFromHistory(fileId, event) {
    if (event) event.stopPropagation();
    if (!confirm('Are you sure you want to delete this document and all its chat history?')) return;

    try {
        const res = await fetch(`${API_URL}/history/${fileId}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed to delete document');

        showToast('Document and history deleted', 'success');
        
        if (state.fileId === fileId) {
            closeDocument();
        }
        
        loadHistoryList();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function handleClearAllHistory() {
    if (!confirm('Are you sure you want to clear ALL document and chat history? This cannot be undone.')) return;

    try {
        const res = await fetch(`${API_URL}/history/clear`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed to clear history');

        showToast('All document history cleared', 'success');
        hideHistoryModal();
        closeDocument();
        loadHistoryList();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function handleClearActiveChat() {
    if (!state.fileId) return;
    if (!confirm('Clear the conversation history for this document?')) return;

    try {
        await fetch(`${API_URL}/history/chat/${state.fileId}`, { method: 'DELETE' });
        dom.chatHistory.innerHTML = '';
        renderWelcomeBanner();
        showToast('Chat history cleared for this document', 'success');
        loadHistoryList();
    } catch (err) {
        showToast('Failed to clear chat history', 'error');
    }
}

// -------------------------------------------------------------------------- //
// File Upload & Pipeline Status
// -------------------------------------------------------------------------- //
async function handleFileUpload(file) {
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
        return showToast('Please select a valid PDF file.', 'error');
    }
    if (file.size > 50 * 1024 * 1024) {
        return showToast('File size exceeds the 50 MB limit.', 'error');
    }

    setUploadState(true, 'Uploading Document…', file.name, 'stepUpload');

    try {
        const form = new FormData();
        form.append('file', file);
        
        const res = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: form,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Upload failed');

        state.fileId = data.file_id;
        setUploadState(true, 'Reading & Extracting Pages…', 'Extracting text with PyMuPDF', 'stepExtract');
        startStatusPolling();
    } catch (err) {
        setUploadState(false);
        state.fileId = null;
        showToast(err.message, 'error');
    }
}

function startStatusPolling() {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(async () => {
        try {
            const res = await fetch(`${API_URL}/status/${state.fileId}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Status check failed');

            if (data.status === 'ready') {
                setPipelineStep('stepReady');
                setTimeout(() => {
                    enterActiveWorkspace(data);
                    loadHistoryList();
                }, 400);
            } else if (data.status === 'error') {
                throw new Error(data.error || 'Document processing encountered an error');
            } else {
                if (data.stage === 'extracting') {
                    setUploadState(true, 'Extracting Page Content…', 'Reading text structures and pages', 'stepExtract');
                } else if (data.stage === 'indexing') {
                    setUploadState(true, 'Generating Vector Index…', 'Embedding 180-word overlapping chunks with MiniLM', 'stepIndex');
                }
                startStatusPolling();
            }
        } catch (err) {
            setUploadState(false);
            state.fileId = null;
            showToast(err.message, 'error');
        }
    }, 600);
}

function setUploadState(processing, title, detail, activeStepId) {
    dom.uploadContent.classList.toggle('hidden', processing);
    dom.uploadProgress.classList.toggle('hidden', !processing);
    if (processing) {
        dom.progressStageTitle.textContent = title || 'Processing…';
        dom.progressStageDetail.textContent = detail || '';
        if (activeStepId) setPipelineStep(activeStepId);
    }
}

function setPipelineStep(stepId) {
    const steps = ['stepUpload', 'stepExtract', 'stepIndex', 'stepReady'];
    let reached = true;
    steps.forEach(s => {
        const el = dom[s];
        if (!el) return;
        el.classList.remove('active', 'completed');
        if (s === stepId) {
            el.classList.add('active');
            reached = false;
        } else if (reached) {
            el.classList.add('completed');
        }
    });
}

// -------------------------------------------------------------------------- //
// Active Workspace Setup
// -------------------------------------------------------------------------- //
async function enterActiveWorkspace(status) {
    dom.initialView.classList.add('hidden');
    dom.activeWorkspace.classList.remove('hidden');
    dom.navNewDocBtn.classList.remove('hidden');
    dom.headerDocInfo.classList.remove('hidden');

    const filename = status.filename || 'document.pdf';
    dom.docTitle.textContent = filename;
    dom.docTitle.title = filename;
    dom.headerDocName.textContent = filename;

    dom.statPages.textContent = status.pages ?? '0';
    dom.statWords.textContent = (status.words ?? 0).toLocaleString();
    dom.statChunks.textContent = status.chunks ?? '0';

    state.totalPages = Math.max(1, status.pages || 1);
    state.currentPage = 1;
    dom.readerPageInput.max = state.totalPages;
    dom.readerTotalPages.textContent = state.totalPages;

    // Reset chat
    dom.chatHistory.innerHTML = '';
    renderWelcomeBanner();

    // Fetch rich document data, keywords & suggested questions concurrently
    loadDocumentReaderData();
    loadKeywords();
    loadSuggestedQuestions();

    switchTab('chatTab');
    dom.chatInput.focus();
    showToast(`Document "${filename}" ready!`, 'success');
}

function closeDocument() {
    clearTimeout(state.pollTimer);
    if (state.speechSynthUtterance) {
        window.speechSynthesis.cancel();
    }
    state.fileId = null;
    state.docData = null;
    state.isBusy = false;
    if (state.abortController) {
        state.abortController.abort();
    }

    dom.activeWorkspace.classList.add('hidden');
    dom.initialView.classList.remove('hidden');
    dom.navNewDocBtn.classList.add('hidden');
    dom.headerDocInfo.classList.add('hidden');
    setUploadState(false);
    loadHistoryList();
}

function renderWelcomeBanner() {
    const div = document.createElement('div');
    div.className = 'chat-msg ai';
    div.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-bolt-lightning"></i></div>
        <div class="msg-body-wrapper">
            <div class="msg-bubble markdown-body">
                <h3>Document Indexed &amp; Ready</h3>
                <p>Welcome! You can now explore your document with AI:</p>
                <ul>
                    <li>⚡ <strong>Quick Summaries:</strong> Choose a format on the left sidebar (Bullets, Overview, Deep Dive, Insights, or FAQ).</li>
                    <li>💬 <strong>Grounded Chat:</strong> Ask any specific question — answers cite exact page numbers <code>(p. N)</code>.</li>
                    <li>💾 <strong>Persistent History:</strong> All your questions and AI answers are automatically saved to SQLite.</li>
                    <li>📖 <strong>Document Reader:</strong> Read pages side-by-side in the Reader tab.</li>
                </ul>
            </div>
        </div>
    `;
    dom.chatHistory.appendChild(div);
}

// -------------------------------------------------------------------------- //
// Tabs & Layout
// -------------------------------------------------------------------------- //
function switchTab(tabId) {
    state.activeTab = tabId;
    dom.navTabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tabId));
    dom.tabPanes.forEach(p => p.classList.toggle('hidden', p.id !== tabId));
    
    if (tabId === 'readerTab') {
        renderCurrentPage();
    } else if (tabId === 'topicsTab') {
        renderTopicsGrid();
    }
    closeSidebarDrawer();
}

function toggleSidebarDrawer() {
    dom.workspaceSidebar.classList.toggle('drawer-open');
    dom.sidebarBackdrop.classList.toggle('hidden', !dom.workspaceSidebar.classList.contains('drawer-open'));
}

function closeSidebarDrawer() {
    dom.workspaceSidebar.classList.remove('drawer-open');
    dom.sidebarBackdrop.classList.add('hidden');
}

// -------------------------------------------------------------------------- //
// Document Reader & Search
// -------------------------------------------------------------------------- //
async function loadDocumentReaderData() {
    try {
        const res = await fetch(`${API_URL}/document/${state.fileId}`);
        const data = await res.json();
        if (res.ok) {
            state.docData = data;
            state.totalPages = Math.max(1, data.pages?.length || data.total_pages || 1);
            dom.readerPageInput.max = state.totalPages;
            dom.readerTotalPages.textContent = state.totalPages;
        }
    } catch (err) {
        console.warn('Reader data prefetch failed:', err);
    }
}

function setReaderPage(page) {
    page = Math.max(1, Math.min(state.totalPages, page));
    state.currentPage = page;
    dom.readerPageInput.value = page;
    renderCurrentPage();
}

function navigatePage(delta) {
    setReaderPage(state.currentPage + delta);
}

function renderCurrentPage() {
    if (!state.docData || !state.docData.pages) {
        dom.pageTextContent.textContent = 'Loading page text…';
        return;
    }

    const pageItem = state.docData.pages.find(p => p.page === state.currentPage) || 
                     state.docData.pages[state.currentPage - 1];

    dom.pageCardBadge.textContent = `Page ${state.currentPage} of ${state.totalPages}`;
    
    if (pageItem) {
        dom.pageWordCount.textContent = `${pageItem.words || pageItem.text.split(/\s+/).length} words`;
        
        const query = dom.readerSearchInput.value.trim();
        if (query) {
            dom.pageTextContent.innerHTML = highlightSearchText(pageItem.text, query);
        } else {
            dom.pageTextContent.textContent = pageItem.text;
        }
    } else {
        dom.pageWordCount.textContent = '0 words';
        dom.pageTextContent.textContent = 'No text content available for this page.';
    }
}

function handleReaderSearch() {
    const query = dom.readerSearchInput.value.trim();
    dom.clearSearchBtn.classList.toggle('hidden', !query);
    renderCurrentPage();
}

function highlightSearchText(text, query) {
    if (!query) return escapeHtml(text);
    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${escaped})`, 'gi');
    return escapeHtml(text).replace(regex, '<mark class="reader-highlight">$1</mark>');
}

// -------------------------------------------------------------------------- //
// Keywords Cloud & Topic Explorer
// -------------------------------------------------------------------------- //
async function loadKeywords() {
    try {
        const res = await fetch(`${API_URL}/keywords/${state.fileId}?top=20`);
        const data = await res.json();
        if (res.ok && data.keywords) {
            state.keywords = data.keywords;
            dom.kwCountBadge.textContent = data.keywords.length;
            renderKeywordsCloud(data.keywords);
        }
    } catch (err) {
        console.warn('Keywords fetch failed:', err);
    }
}

function renderKeywordsCloud(keywords) {
    if (!keywords.length) {
        dom.keywordsList.innerHTML = '<span style="color:var(--text-muted);font-size:12px">No topics found.</span>';
        return;
    }

    dom.keywordsList.innerHTML = '';
    keywords.forEach(kw => {
        const pill = document.createElement('button');
        pill.className = 'topic-pill';
        pill.textContent = kw.term;
        pill.title = `Relevance Score: ${Math.round(kw.score * 100)}%`;
        pill.addEventListener('click', () => {
            switchTab('chatTab');
            dom.chatInput.value = `Explain how "${kw.term}" is discussed in this document.`;
            autoResizeTextarea();
            handleChatSubmit();
        });
        dom.keywordsList.appendChild(pill);
    });
}

function renderTopicsGrid() {
    if (!state.keywords.length) {
        dom.topicsExplorerGrid.innerHTML = `
            <div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--text-muted)">
                <i class="fa-solid fa-spinner fa-spin" style="font-size:24px"></i>
                <p style="margin-top:12px">Extracting key topics from document…</p>
            </div>
        `;
        return;
    }

    dom.topicsExplorerGrid.innerHTML = state.keywords.map((kw, i) => `
        <div class="topic-card">
            <div class="topic-card-header">
                <span class="topic-rank-badge">#${i + 1}</span>
                <span class="topic-score-badge">${Math.round(kw.score * 100)}% score</span>
            </div>
            <h4 class="topic-card-title">${escapeHtml(kw.term)}</h4>
            <p class="topic-card-desc">Extracted via TF-IDF statistical scoring.</p>
            <button class="btn btn-outline btn-sm btn-full" onclick="askAboutTopic('${escapeHtml(kw.term)}')">
                <i class="fa-solid fa-comment-dots"></i> Explore Topic
            </button>
        </div>
    `).join('');
}

window.askAboutTopic = function(term) {
    switchTab('chatTab');
    dom.chatInput.value = `What does the document say about "${term}"? Give specific details and page references.`;
    autoResizeTextarea();
    handleChatSubmit();
};

// -------------------------------------------------------------------------- //
// Suggested Questions
// -------------------------------------------------------------------------- //
async function loadSuggestedQuestions() {
    dom.suggestionChips.innerHTML = '';
    try {
        const res = await fetch(`${API_URL}/suggest/${state.fileId}`);
        const data = await res.json();
        if (!res.ok || !data.questions || !data.questions.length) {
            dom.suggestionChips.classList.add('hidden');
            return;
        }

        dom.suggestionChips.innerHTML = '';
        data.questions.forEach(q => {
            const chip = document.createElement('button');
            chip.className = 'suggestion-chip';
            chip.innerHTML = `<i class="fa-solid fa-sparkles"></i> <span>${escapeHtml(q)}</span>`;
            chip.addEventListener('click', () => {
                dom.chatInput.value = q;
                autoResizeTextarea();
                handleChatSubmit();
            });
            dom.suggestionChips.appendChild(chip);
        });
        dom.suggestionChips.classList.remove('hidden');
    } catch {
        dom.suggestionChips.classList.add('hidden');
    }
}

// -------------------------------------------------------------------------- //
// Interrupt / Stop Generation Engine
// -------------------------------------------------------------------------- //
function setBusyState(busy, thinkingLabel = 'Thinking') {
    state.isBusy = busy;
    dom.sendBtn.disabled = busy;
    dom.stopGenBtn.classList.toggle('hidden', !busy);
    document.querySelectorAll('.summary-action-btn').forEach(b => b.disabled = busy);

    if (busy) {
        state.abortController = new AbortController();
    } else {
        state.abortController = null;
    }
}

function interruptGeneration() {
    if (!state.isBusy) return;
    
    if (state.abortController) {
        state.abortController.abort();
    }
    
    if (state.fileId) {
        fetch(`${API_URL}/cancel/${state.fileId}`, { method: 'POST' }).catch(() => {});
    }

    setBusyState(false);
    
    const lastMsg = dom.chatHistory.querySelector('.chat-msg.ai:last-child');
    if (lastMsg && lastMsg.querySelector('.thinking-box')) {
        const bubble = lastMsg.querySelector('.msg-bubble');
        bubble.innerHTML = `
            <div class="interrupted-tag">
                <i class="fa-solid fa-circle-pause"></i> Generation stopped by user.
            </div>
        `;
    }

    showToast('Generation interrupted', 'error');
    dom.chatInput.focus();
}

// -------------------------------------------------------------------------- //
// AI Summary Generator
// -------------------------------------------------------------------------- //
async function triggerSummary(length, btnEl) {
    if (!state.fileId || state.isBusy) return;
    
    switchTab('chatTab');
    setBusyState(true, 'Synthesizing Summary');

    const titles = {
        short: 'Quick Bullets',
        medium: 'Executive Overview',
        detailed: 'Detailed Report',
        findings: 'Key Insights & Data',
        faq: 'Essential FAQ',
    };

    appendUserMessage(`Generate a ${titles[length] || 'Summary'} for this document.`);
    const aiMsgEl = appendThinkingMessage(`Synthesizing ${titles[length] || 'summary'}…`);

    try {
        const res = await fetch(`${API_URL}/summary`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_id: state.fileId, length }),
            signal: state.abortController.signal,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Summary generation failed');

        renderAiResponse(aiMsgEl, data.summary, {
            footer: `${data.model || 'AI'}${data.cached ? ' · cached' : ''}`,
        });
        
        // Refresh history to update message counts
        loadHistoryList();
    } catch (err) {
        if (err.name === 'AbortError') return;
        renderErrorMessage(aiMsgEl, err.message);
    } finally {
        setBusyState(false);
        scrollChatToBottom();
    }
}

// -------------------------------------------------------------------------- //
// Chat Engine & Grounded Q&A
// -------------------------------------------------------------------------- //
async function handleChatSubmit() {
    const question = dom.chatInput.value.trim();
    if (!question || !state.fileId || state.isBusy) return;

    if (question.startsWith('/')) {
        dom.chatInput.value = '';
        autoResizeTextarea();
        handleSlashCommand(question);
        return;
    }

    setBusyState(true, 'Searching & Reasoning');
    dom.chatInput.value = '';
    autoResizeTextarea();

    appendUserMessage(question);
    const aiMsgEl = appendThinkingMessage('Retrieving document passages and generating answer…');

    try {
        const res = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_id: state.fileId, question }),
            signal: state.abortController.signal,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to generate response');

        renderAiResponse(aiMsgEl, data.answer, {
            sources: data.sources,
            footer: `${data.model || 'OpenRouter'} · ${data.chunks_used} passage${data.chunks_used === 1 ? '' : 's'} verified`,
        });

        // Refresh history to update message counts
        loadHistoryList();
    } catch (err) {
        if (err.name === 'AbortError') return;
        renderErrorMessage(aiMsgEl, err.message);
    } finally {
        setBusyState(false);
        scrollChatToBottom();
        dom.chatInput.focus();
    }
}

function handleSlashCommand(cmd) {
    const cleaned = cmd.toLowerCase().trim();
    if (cleaned.startsWith('/summary')) triggerSummary('short');
    else if (cleaned.startsWith('/overview')) triggerSummary('medium');
    else if (cleaned.startsWith('/detailed')) triggerSummary('detailed');
    else if (cleaned.startsWith('/findings')) triggerSummary('findings');
    else if (cleaned.startsWith('/faq')) triggerSummary('faq');
    else if (cleaned.startsWith('/clear')) handleClearActiveChat();
    else {
        showToast(`Unknown command: ${cmd}`, 'error');
    }
}

function handleInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        dom.slashHelper.classList.add('hidden');
        handleChatSubmit();
    }
}

function handleChatInputChanges() {
    autoResizeTextarea();
    const val = dom.chatInput.value;
    if (val.startsWith('/') && val.length < 10) {
        dom.slashHelper.classList.remove('hidden');
    } else {
        dom.slashHelper.classList.add('hidden');
    }
}

function autoResizeTextarea() {
    dom.chatInput.style.height = 'auto';
    dom.chatInput.style.height = Math.min(dom.chatInput.scrollHeight, 140) + 'px';
}

// -------------------------------------------------------------------------- //
// Message Rendering & Interactive Citations
// -------------------------------------------------------------------------- //
function appendUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'chat-msg user';
    div.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-user"></i></div>
        <div class="msg-body-wrapper">
            <div class="msg-bubble">${escapeHtml(text)}</div>
        </div>
    `;
    dom.chatHistory.appendChild(div);
    scrollChatToBottom();
    return div;
}

function appendThinkingMessage(label = 'Thinking…') {
    const div = document.createElement('div');
    div.className = 'chat-msg ai';
    div.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-bolt-lightning"></i></div>
        <div class="msg-body-wrapper">
            <div class="msg-bubble">
                <div class="thinking-box">
                    <div class="typing-wave"><span></span><span></span><span></span></div>
                    <span style="font-size:13px;color:var(--text-muted);margin-left:8px">${escapeHtml(label)}</span>
                </div>
            </div>
        </div>
    `;
    dom.chatHistory.appendChild(div);
    scrollChatToBottom();
    return div;
}

function renderAiResponse(msgEl, markdownText, { sources = null, footer = '' } = {}) {
    const bodyWrapper = msgEl.querySelector('.msg-body-wrapper');
    const bubble = msgEl.querySelector('.msg-bubble');
    
    bubble.classList.add('markdown-body');
    const htmlContent = marked.parse(markdownText || '');
    
    const enhancedHtml = enrichPageCitations(htmlContent, sources);
    bubble.innerHTML = enhancedHtml;

    bubble.querySelectorAll('.citation-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const pageNum = parseInt(chip.dataset.page, 10);
            const snippet = chip.dataset.snippet || `Referenced passage on Page ${pageNum}`;
            showPassageModal(pageNum, snippet, chip.dataset.score || '');
        });
    });

    if (sources && sources.length) {
        const sourcesDeck = document.createElement('div');
        sourcesDeck.className = 'msg-sources-deck';
        sourcesDeck.innerHTML = `
            <div class="sources-deck-title"><i class="fa-solid fa-book-bookmark"></i> Grounded Citations</div>
            <div class="sources-chip-row"></div>
        `;
        const row = sourcesDeck.querySelector('.sources-chip-row');
        sources.forEach(src => {
            const btn = document.createElement('button');
            btn.className = 'source-item-btn';
            btn.type = 'button';
            btn.innerHTML = `<i class="fa-solid fa-bookmark"></i> Page ${src.page} <small>(${Math.round(src.score * 100)}%)</small>`;
            btn.title = src.snippet;
            btn.addEventListener('click', () => showPassageModal(src.page, src.snippet, src.score));
            row.appendChild(btn);
        });
        bubble.appendChild(sourcesDeck);
    }

    const actionsBar = document.createElement('div');
    actionsBar.className = 'msg-actions-bar';
    actionsBar.innerHTML = `
        <button class="msg-act-btn btn-copy" title="Copy response to clipboard"><i class="fa-regular fa-copy"></i> Copy</button>
        <button class="msg-act-btn btn-speak" title="Read response aloud"><i class="fa-solid fa-volume-high"></i> Listen</button>
        <button class="msg-act-btn btn-view-reader" title="Open in Document Reader"><i class="fa-solid fa-book-open"></i> Read Doc</button>
        ${footer ? `<span class="msg-model-tag">${footer}</span>` : ''}
    `;

    actionsBar.querySelector('.btn-copy').addEventListener('click', async (e) => {
        await navigator.clipboard.writeText(markdownText);
        e.currentTarget.innerHTML = '<i class="fa-solid fa-check" style="color:var(--success)"></i> Copied!';
        setTimeout(() => {
            e.currentTarget.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
        }, 2000);
    });

    const speakBtn = actionsBar.querySelector('.btn-speak');
    speakBtn.addEventListener('click', () => toggleSpeech(markdownText, speakBtn));

    actionsBar.querySelector('.btn-view-reader').addEventListener('click', () => {
        switchTab('readerTab');
    });

    bodyWrapper.appendChild(actionsBar);
    scrollChatToBottom();
}

function renderErrorMessage(msgEl, message) {
    const bubble = msgEl.querySelector('.msg-bubble');
    bubble.innerHTML = `
        <div style="color:#f87171;display:flex;align-items:center;gap:8px">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <span>${escapeHtml(message)}</span>
        </div>
    `;
    scrollChatToBottom();
}

function enrichPageCitations(html, sources) {
    return html.replace(/\(p\.\s*(\d+)\)/gi, (match, page) => {
        const matchingSrc = sources ? sources.find(s => String(s.page) === String(page)) : null;
        const snippet = matchingSrc ? escapeHtml(matchingSrc.snippet) : '';
        const score = matchingSrc ? Math.round(matchingSrc.score * 100) + '%' : '';
        return `<button class="citation-chip" data-page="${page}" data-snippet="${snippet}" data-score="${score}" title="Click to view Page ${page} excerpt"><i class="fa-solid fa-bookmark"></i> p. ${page}</button>`;
    });
}

function scrollChatToBottom() {
    if (!dom.chatHistory) return;
    requestAnimationFrame(() => {
        dom.chatHistory.scrollTop = dom.chatHistory.scrollHeight;
        setTimeout(() => {
            if (dom.chatHistory) dom.chatHistory.scrollTop = dom.chatHistory.scrollHeight;
        }, 60);
    });
}

// -------------------------------------------------------------------------- //
// Passage Inspector Modal
// -------------------------------------------------------------------------- //
function showPassageModal(page, snippet, score) {
    dom.modalPageTitle.textContent = `Passage Citation (Page ${page})`;
    dom.modalScoreBadge.textContent = score ? `Relevance: ${score}` : `Page ${page}`;
    dom.modalExcerptText.textContent = snippet || 'Excerpt unavailable.';
    dom.modalJumpReaderBtn.dataset.page = page;
    dom.passageModal.classList.remove('hidden');
}

function hidePassageModal() {
    dom.passageModal.classList.add('hidden');
}

// -------------------------------------------------------------------------- //
// Speech Synthesis (Read Aloud)
// -------------------------------------------------------------------------- //
function toggleSpeech(text, btn) {
    if (!('speechSynthesis' in window)) {
        return showToast('Text-to-speech is not supported in this browser.', 'error');
    }

    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        btn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Listen';
        return;
    }

    const cleanText = text.replace(/[#*_`~[\]()]/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onend = () => {
        btn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Listen';
    };
    utterance.onerror = () => {
        btn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Listen';
    };

    state.speechSynthUtterance = utterance;
    window.speechSynthesis.speak(utterance);
    btn.innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
}

// -------------------------------------------------------------------------- //
// Exporter Suite
// -------------------------------------------------------------------------- //
async function handleExport(fmt) {
    if (!state.fileId) return showToast('No active document to export.', 'error');

    if (fmt === 'md') {
        window.open(`${API_URL}/export/${state.fileId}?format=md`, '_blank');
        showToast('Exporting summary as Markdown…', 'success');
    } else if (fmt === 'txt') {
        window.open(`${API_URL}/export/${state.fileId}?format=txt`, '_blank');
        showToast('Exporting summary as Plain Text…', 'success');
    } else if (fmt === 'copy') {
        try {
            const res = await fetch(`${API_URL}/export/${state.fileId}?format=md`);
            const text = await res.text();
            await navigator.clipboard.writeText(text);
            showToast('Summary copied to clipboard!', 'success');
        } catch {
            showToast('Failed to copy summary.', 'error');
        }
    } else if (fmt === 'print') {
        window.print();
    }
}

// -------------------------------------------------------------------------- //
// Date/Time Formatting & Helpers
// -------------------------------------------------------------------------- //
function formatTimeAgo(timestamp) {
    if (!timestamp) return 'recently';
    const d = new Date(timestamp.replace(' ', 'T') + 'Z');
    const now = new Date();
    const diffMs = Math.max(0, now - d);
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHrs = Math.floor(diffMin / 60);
    const diffDays = Math.floor(diffHrs / 24);

    if (diffSec < 60) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHrs < 24) return `${diffHrs}h ago`;
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function formatDateString(timestamp) {
    if (!timestamp) return 'Recent';
    try {
        const d = new Date(timestamp.replace(' ', 'T') + 'Z');
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch {
        return timestamp;
    }
}

// -------------------------------------------------------------------------- //
// Toast Notifications & Helpers
// -------------------------------------------------------------------------- //
let toastTimer = null;
function showToast(message, type = 'info', duration = 3800) {
    dom.toastMessage.textContent = message;
    dom.toast.className = `toast-card ${type}`;
    
    if (type === 'success') {
        dom.toastIcon.className = 'fa-solid fa-circle-check toast-icon';
    } else if (type === 'error') {
        dom.toastIcon.className = 'fa-solid fa-triangle-exclamation toast-icon';
    } else {
        dom.toastIcon.className = 'fa-solid fa-circle-info toast-icon';
    }

    dom.toast.classList.remove('hidden');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(hideToast, duration);
}

function hideToast() {
    dom.toast.classList.add('hidden');
}

function escapeHtml(str) {
    return (str || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
