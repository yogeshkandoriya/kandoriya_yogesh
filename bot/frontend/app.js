/**
 * Summary AI — Next-Generation Frontend Controller
 * Complete responsive document workspace with grounded RAG,
 * interrupt/abort capability, citation inspection, reader, and multi-format exporter.
 */

// -------------------------------------------------------------------------- //
// Configuration & Environment
// -------------------------------------------------------------------------- //
/* Same-origin whenever the page itself is served over HTTP(S) — covers dev
   (:8000) and any host (huggingface.co, Railway, Render...). file:// fallback
   points at a locally running backend. */
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

    // Sidebar Data
    docTitle: document.getElementById('docTitle'),
    statPages: document.getElementById('statPages'),
    statWords: document.getElementById('statWords'),
    statChunks: document.getElementById('statChunks'),
    keywordsList: document.getElementById('keywordsList'),
    kwCountBadge: document.getElementById('kwCountBadge'),
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

    // Modals
    passageModal: document.getElementById('passageModal'),
    modalPageTitle: document.getElementById('modalPageTitle'),
    modalScoreBadge: document.getElementById('modalScoreBadge'),
    modalExcerptText: document.getElementById('modalExcerptText'),
    closePassageModal: document.getElementById('closePassageModal'),
    modalJumpReaderBtn: document.getElementById('modalJumpReaderBtn'),
    modalAskPassageBtn: document.getElementById('modalAskPassageBtn'),

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
        if (state.fileId && confirm('Return to home screen? Your current document remains active.')) {
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
    dom.clearChatBtn.addEventListener('click', () => {
        if (confirm('Clear the conversation history for this document?')) {
            dom.chatHistory.innerHTML = '';
            renderWelcomeBanner();
            showToast('Chat history cleared', 'success');
        }
    });

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
        
        // Populate diagnostics
        dom.diagPrimaryModel.textContent = data.model || 'nvidia/nemotron-3-super-120b-a12b:free';
        dom.diagFallbacks.innerHTML = (data.fallbacks || []).map(f => `<div class="code-badge" style="margin-top:4px">${f}</div>`).join('') || 'None';
        dom.diagKeyStatus.textContent = data.api_key_configured ? 'Configured & Active' : 'Missing (Check .env)';
        dom.diagKeyStatus.style.color = data.api_key_configured ? 'var(--success)' : 'var(--danger)';
    } catch {
        dom.modelNameText.textContent = 'server offline';
        dom.serverDot.className = 'status-indicator offline';
        showToast('Cannot connect to backend server on port 8000', 'error');
    }
}

function showDiagnosticsModal() {
    dom.diagnosticsModal.classList.remove('hidden');
}

function hideDiagnosticsModal() {
    dom.diagnosticsModal.classList.add('hidden');
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
                setTimeout(() => enterActiveWorkspace(data), 400);
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
                    <li>📖 <strong>Document Reader:</strong> Read pages side-by-side in the Reader tab.</li>
                    <li>⏹ <strong>Interrupt Generation:</strong> Stop any response instantly with the Stop button or <kbd>Esc</kbd>.</li>
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
    try {
        const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escaped})`, 'gi');
        return text.replace(regex, '<span class="highlight-match">$1</span>');
    } catch {
        return text;
    }
}

// -------------------------------------------------------------------------- //
// Keywords & Topic Explorer
// -------------------------------------------------------------------------- //
async function loadKeywords() {
    dom.keywordsList.innerHTML = '<div class="kw-skeleton-pill"></div><div class="kw-skeleton-pill"></div><div class="kw-skeleton-pill"></div>';
    try {
        const res = await fetch(`${API_URL}/keywords/${state.fileId}?top=18`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Keywords failed');
        
        state.keywords = data.keywords || [];
        dom.kwCountBadge.textContent = state.keywords.length;
        dom.keywordsList.innerHTML = '';

        if (!state.keywords.length) {
            dom.keywordsList.innerHTML = '<span style="font-size:11px;color:var(--text-muted)">No topics extracted</span>';
            return;
        }

        state.keywords.forEach(k => {
            const btn = document.createElement('button');
            btn.className = 'topic-tag';
            btn.type = 'button';
            btn.innerHTML = `<i class="fa-solid fa-tag"></i> <span>${k.term}</span>`;
            btn.title = `Score: ${k.score} · Ask about "${k.term}"`;
            btn.addEventListener('click', () => {
                switchTab('chatTab');
                dom.chatInput.value = `Explain the significance of "${k.term}" in this document.`;
                autoResizeTextarea();
                handleChatSubmit();
            });
            dom.keywordsList.appendChild(btn);
        });
    } catch {
        dom.keywordsList.innerHTML = '<span style="font-size:11px;color:var(--text-muted)">Topics unavailable</span>';
    }
}

function renderTopicsGrid() {
    dom.topicsExplorerGrid.innerHTML = '';
    if (!state.keywords.length) {
        dom.topicsExplorerGrid.innerHTML = '<p style="color:var(--text-muted)">No topic data loaded yet.</p>';
        return;
    }

    const maxScore = Math.max(...state.keywords.map(k => k.score), 0.01);

    state.keywords.forEach((k, idx) => {
        const pct = Math.round((k.score / maxScore) * 100);
        const card = document.createElement('div');
        card.className = 'topic-insight-card';
        card.innerHTML = `
            <div class="card-top">
                <strong>#${idx + 1} ${k.term}</strong>
                <span class="relevance-badge">${pct}% Importance</span>
            </div>
            <div class="topic-score-bar">
                <div class="topic-score-fill" style="width: ${pct}%"></div>
            </div>
            <p style="font-size:12px;color:var(--text-secondary)">Click to analyze how this topic is referenced in the document.</p>
        `;
        card.addEventListener('click', () => {
            switchTab('chatTab');
            dom.chatInput.value = `Provide a comprehensive analysis of "${k.term}" based on this document.`;
            autoResizeTextarea();
            handleChatSubmit();
        });
        dom.topicsExplorerGrid.appendChild(card);
    });
}

// -------------------------------------------------------------------------- //
// Suggested Questions
// -------------------------------------------------------------------------- //
async function loadSuggestedQuestions() {
    try {
        const res = await fetch(`${API_URL}/suggest/${state.fileId}`);
        const data = await res.json();
        if (!res.ok || !data.questions?.length) return;

        dom.suggestionChips.innerHTML = '';
        data.questions.forEach(q => {
            const chip = document.createElement('button');
            chip.className = 'suggestion-pill';
            chip.type = 'button';
            chip.innerHTML = `<i class="fa-regular fa-circle-question"></i> ${q}`;
            chip.addEventListener('click', () => {
                if (state.isBusy) return;
                dom.chatInput.value = q;
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
    
    // Notify server of cancellation if desired
    if (state.fileId) {
        fetch(`${API_URL}/cancel/${state.fileId}`, { method: 'POST' }).catch(() => {});
    }

    setBusyState(false);
    
    // Find active thinking bubble and mark interrupted
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
    } catch (err) {
        if (err.name === 'AbortError') return; // Handled by interruptGeneration
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

    // Check for slash commands
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
    } catch (err) {
        if (err.name === 'AbortError') return; // Handled by interruptGeneration
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
    else if (cleaned.startsWith('/clear')) {
        dom.chatHistory.innerHTML = '';
        renderWelcomeBanner();
        showToast('Chat history cleared', 'success');
    } else {
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
    
    // Format markdown
    bubble.classList.add('markdown-body');
    const htmlContent = marked.parse(markdownText || '');
    
    // Interpret page citations (p. N) into interactive badges
    const enhancedHtml = enrichPageCitations(htmlContent, sources);
    bubble.innerHTML = enhancedHtml;

    // Attach citation chip click listeners
    bubble.querySelectorAll('.citation-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const pageNum = parseInt(chip.dataset.page, 10);
            const snippet = chip.dataset.snippet || `Referenced passage on Page ${pageNum}`;
            showPassageModal(pageNum, snippet, chip.dataset.score || '');
        });
    });

    // Render Grounded Sources Section if present
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

    // Message Actions Toolbar (Copy, Read aloud, Jump to Reader)
    const actionsBar = document.createElement('div');
    actionsBar.className = 'msg-actions-bar';
    actionsBar.innerHTML = `
        <button class="msg-act-btn btn-copy" title="Copy response to clipboard"><i class="fa-regular fa-copy"></i> Copy</button>
        <button class="msg-act-btn btn-speak" title="Read response aloud"><i class="fa-solid fa-volume-high"></i> Listen</button>
        <button class="msg-act-btn btn-view-reader" title="Open in Document Reader"><i class="fa-solid fa-book-open"></i> Read Doc</button>
        ${footer ? `<span class="msg-model-tag">${footer}</span>` : ''}
    `;

    // Copy Action
    actionsBar.querySelector('.btn-copy').addEventListener('click', async (e) => {
        await navigator.clipboard.writeText(markdownText);
        e.currentTarget.innerHTML = '<i class="fa-solid fa-check" style="color:var(--success)"></i> Copied!';
        setTimeout(() => {
            e.currentTarget.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
        }, 2000);
    });

    // Speech / Text to Speech Action
    const speakBtn = actionsBar.querySelector('.btn-speak');
    speakBtn.addEventListener('click', () => toggleSpeech(markdownText, speakBtn));

    // Jump to Reader Action
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

/**
 * Parses markdown text to replace citations like (p. 2) or [Page 2] with interactive buttons.
 */
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

    // Clean markdown for speech
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
