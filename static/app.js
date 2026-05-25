/**
 * OnCall Agent System - Frontend Application
 * Dark Cyber Ops Theme
 */

// ==================== State ====================
const STATE = {
    currentSessionId: 'session_' + Date.now(),
    currentPage: 'chat',
    isStreaming: false,
    agentMode: 'auto',
};

// ==================== DOM Helpers ====================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function getDom() {
    return {
        chatMessages: $('#chatMessages'),
        ragMessages: $('#ragMessages'),
        aiopsMessages: $('#aiopsMessages'),
        messageInput: $('#messageInput'),
        ragInput: $('#ragInput'),
        sendBtn: $('#sendButton'),
        ragSendBtn: $('#ragSendBtn'),
        newChatBtn: $('#newChatBtn'),
        navBtns: $$('.nav-btn'),
        pages: $$('.page'),
        loadingOverlay: $('#loadingOverlay'),
        loadingText: $('#loadingText'),
        agentModeSelect: $('#agentModeSelect'),
        aiopsStartBtn: $('#aiopsStartBtn'),
        diagStatusBar: $('#diagStatusBar'),
        chatHistoryList: $('#chatHistoryList'),
        uploadBox: $('#uploadBox'),
        fileInput: $('#fileInput'),
        docsStatus: $('#docsStatus'),
        indexDocsBtn: $('#indexDocsBtn'),
        indexAiopsBtn: $('#indexAiopsBtn'),
        evalGenerateBtn: $('#evalGenerateBtn'),
        evalRunBtn: $('#evalRunBtn'),
        evalRefreshBtn: $('#evalRefreshBtn'),
        evalMetrics: $('#evalMetrics'),
        evalDetails: $('#evalDetails'),
        chatCharCount: $('#chatCharCount'),
        ragCharCount: $('#ragCharCount'),
    };
}

// ==================== SVG Icons ====================
const AI_AVATAR_SVG = '<svg viewBox="0 0 24 24" fill="none" width="18" height="18"><polygon points="12,2 22,12 12,22 2,12" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>';

// ==================== Utilities ====================
function escapeHtml(text) {
    var d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

function getSessionId() { return STATE.currentSessionId; }

function showLoading(text) {
    var DOM = getDom();
    DOM.loadingText.textContent = text || '处理中...';
    DOM.loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    var el = getDom().loadingOverlay;
    el.style.display = 'none';
}

function showNotification(msg, type) {
    var div = document.createElement('div');
    div.className = 'notification ' + (type || 'info');
    div.textContent = msg;
    document.body.appendChild(div);
    setTimeout(function () {
        div.style.opacity = '0';
        div.style.transition = 'opacity 0.3s';
        setTimeout(function () { div.remove(); }, 300);
    }, 3000);
}

function updateCharCount(input, display) {
    if (!input || !display) return;
    display.textContent = input.value.length + '/' + (input.maxLength || 2000);
}

// ==================== API ====================
async function apiPost(path, body) {
    var resp = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {}),
    });
    return resp.json();
}

async function apiGet(path) {
    var resp = await fetch(path);
    return resp.json();
}

async function apiUpload(file) {
    var fd = new FormData();
    fd.append('file', file);
    var resp = await fetch('/api/upload', { method: 'POST', body: fd });
    return resp.json();
}

// ==================== Message Renderer ====================
function addMessage(container, role, content, extraClass) {
    if (!content) return null;
    var wrapper = document.createElement('div');
    wrapper.className = 'message ' + role + (extraClass ? ' ' + extraClass : '');

    var avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = AI_AVATAR_SVG;

    var cw = document.createElement('div');
    cw.className = 'message-content-wrapper';

    var cd = document.createElement('div');
    cd.className = 'message-content';
    cd.innerHTML = marked.parse(content);

    cw.appendChild(cd);
    if (role === 'assistant') wrapper.appendChild(avatar);
    wrapper.appendChild(cw);
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
    return cd;
}

function addEventMsg(container, cls, text) {
    var d = document.createElement('div');
    d.className = 'event-' + cls;
    d.textContent = text;
    container.appendChild(d);
}

function makeAssistantContainer(container) {
    var wrapper = document.createElement('div');
    wrapper.className = 'message assistant';
    var avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = AI_AVATAR_SVG;
    var cw = document.createElement('div');
    cw.className = 'message-content-wrapper';
    var cd = document.createElement('div');
    cd.className = 'message-content';
    cw.appendChild(cd);
    wrapper.appendChild(avatar);
    wrapper.appendChild(cw);
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
    return { wrapper: wrapper, content: cd };
}

// ==================== Unified Agent Chat ====================
async function sendMessage() {
    var DOM = getDom();
    var input = DOM.messageInput;
    var text = input.value.trim();
    if (!text || STATE.isStreaming) return;
    input.value = '';
    updateCharCount(input, DOM.chatCharCount);

    var container = DOM.chatMessages;
    var welcome = container.querySelector('.welcome-screen');
    if (welcome) welcome.style.display = 'none';

    addMessage(container, 'user', text);

    var ai = makeAssistantContainer(container);
    var contentDiv = ai.content;

    // route hint
    var routeHint = document.createElement('div');
    routeHint.className = 'event-route';
    routeHint.textContent = '⏳ 正在路由...';
    contentDiv.appendChild(routeHint);

    STATE.isStreaming = true;
    DOM.sendBtn.disabled = true;
    contentDiv.closest('.message')?.classList.add('streaming');

    try {
        var resp = await fetch('/api/agent_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: getSessionId(), question: text, mode: STATE.agentMode }),
        });
        var reader = resp.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';
        var fullHtml = '';

        while (true) {
            var { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            var lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (var i = 0; i < lines.length; i++) {
                var line = lines[i];
                if (line.startsWith('data: ')) {
                    try {
                        var evtData = JSON.parse(line.slice(6));
                        handleAgentEvent(contentDiv, evtData, routeHint);
                    } catch (e) { /* skip malformed */ }
                }
            }
        }
    } catch (err) {
        contentDiv.innerHTML = '<p style="color:var(--accent-red)">请求失败: ' + escapeHtml(err.message) + '</p>';
    }

    var msgEl = contentDiv.closest('.message');
    if (msgEl) msgEl.classList.remove('streaming');
    STATE.isStreaming = false;
    DOM.sendBtn.disabled = false;
    container.scrollTop = container.scrollHeight;
    loadHistory();
}

function handleAgentEvent(contentDiv, evt, routeHint) {
    var type = evt.type || '';

    if (type === 'route') {
        if (routeHint) routeHint.textContent = '🔀 路由: ' + (evt.data ? (evt.data.route || '') : '');
        return;
    }

    if (type === 'content') {
        var text = evt.data || '';
        var lastP = contentDiv.querySelector('.ai-text');
        if (lastP) {
            lastP.textContent += text;
        } else {
            var p = document.createElement('p');
            p.className = 'ai-text';
            p.textContent = text;
            contentDiv.appendChild(p);
        }
        if (routeHint && routeHint.parentNode) routeHint.remove();
        return;
    }

    if (type === 'event') {
        var ed = evt.data || {};
        var et = ed.type || '';
        if (et === 'plan') {
            addEventMsg(contentDiv, 'plan', '📋 ' + (ed.message || '计划已制定'));
        } else if (et === 'step_complete') {
            addEventMsg(contentDiv, 'step', '✅ ' + (ed.current_step || ''));
        } else if (et === 'report') {
            addEventMsg(contentDiv, 'report', '📊 诊断报告已生成');
        } else {
            addEventMsg(contentDiv, 'tool', '⚙ ' + (ed.message || ''));
        }
        return;
    }

    if (type === 'complete') {
        var note = document.createElement('div');
        note.className = 'ref-source';
        note.textContent = '✓ 完成';
        contentDiv.appendChild(note);
        return;
    }

    if (type === 'error') {
        addEventMsg(contentDiv, 'tool', '❌ 错误: ' + (evt.data || ''));
    }
}

// ==================== RAG Chat ====================
async function sendRagQuery() {
    var DOM = getDom();
    var input = DOM.ragInput;
    var text = input.value.trim();
    if (!text || STATE.isStreaming) return;
    input.value = '';
    updateCharCount(input, DOM.ragCharCount);

    var container = DOM.ragMessages;
    var welcome = container.querySelector('.welcome-screen');
    if (welcome) welcome.style.display = 'none';

    addMessage(container, 'user', text);
    STATE.isStreaming = true;
    DOM.ragSendBtn.disabled = true;

    var ai = makeAssistantContainer(container);
    var contentDiv = ai.content;
    ai.wrapper.classList.add('streaming');

    try {
        var resp = await fetch('/api/chat_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ Id: getSessionId(), Question: text }),
        });
        var reader = resp.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';
        var fullAnswer = '';

        while (true) {
            var { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            var lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (var i = 0; i < lines.length; i++) {
                var line = lines[i];
                if (line.startsWith('data: ')) {
                    try {
                        var data = JSON.parse(line.slice(6));
                        var t = data.type || '';
                        if (t === 'content') {
                            fullAnswer += data.data || '';
                            contentDiv.innerHTML = marked.parse(fullAnswer);
                        } else if (t === 'search_results') {
                            var ref = document.createElement('div');
                            ref.className = 'ref-source';
                            ref.textContent = '📚 检索到 ' + (data.data || []).length + ' 篇相关文档';
                            contentDiv.appendChild(ref);
                        } else if (t === 'tool_call') {
                            addEventMsg(contentDiv, 'tool', '🔧 工具: ' + ((data.data && data.data.tool) || ''));
                        } else if (t === 'done') {
                            var note = document.createElement('div');
                            note.className = 'ref-source';
                            note.textContent = '✓ 回答完成';
                            contentDiv.appendChild(note);
                        } else if (t === 'error') {
                            contentDiv.innerHTML += '<p style="color:var(--accent-red)">错误: ' + escapeHtml(data.data || '') + '</p>';
                        }
                    } catch (e) { /* skip */ }
                }
            }
        }
    } catch (err) {
        contentDiv.innerHTML = '<p style="color:var(--accent-red)">请求失败: ' + escapeHtml(err.message) + '</p>';
    }

    ai.wrapper.classList.remove('streaming');
    STATE.isStreaming = false;
    DOM.ragSendBtn.disabled = false;
    container.scrollTop = container.scrollHeight;
}

// ==================== AIOps ====================
async function startDiagnose() {
    if (STATE.isStreaming) return;
    var DOM = getDom();
    var container = DOM.aiopsMessages;
    var welcome = container.querySelector('.welcome-screen');
    if (welcome) welcome.style.display = 'none';
    container.innerHTML = '';
    STATE.isStreaming = true;
    DOM.aiopsStartBtn.disabled = true;
    DOM.diagStatusBar.innerHTML = '<span class="diag-dot" style="background:var(--accent-cyan);animation:pulseDot 1.5s infinite"></span> 诊断进行中...';

    var ai = makeAssistantContainer(container);
    ai.wrapper.classList.add('aiops-message');
    var contentDiv = ai.content;

    try {
        var resp = await fetch('/api/aiops', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: getSessionId() }),
        });
        var reader = resp.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';

        while (true) {
            var { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            var lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (var i = 0; i < lines.length; i++) {
                var line = lines[i];
                if (line.startsWith('data: ')) {
                    try {
                        var evt = JSON.parse(line.slice(6));
                        var type = evt.type || '';
                        if (type === 'plan') {
                            addEventMsg(contentDiv, 'plan', '📋 ' + (evt.message || '计划已制定'));
                            DOM.diagStatusBar.innerHTML = '<span class="diag-dot" style="background:var(--accent-amber);animation:pulseDot 1.5s infinite"></span> ' + (evt.message || '');
                        } else if (type === 'step_complete') {
                            addEventMsg(contentDiv, 'step', '✅ ' + (evt.current_step || ''));
                            DOM.diagStatusBar.innerHTML = '<span class="diag-dot" style="background:var(--accent-green);animation:pulseDot 1.5s infinite"></span> ' + (evt.message || evt.current_step || '');
                        } else if (type === 'report') {
                            addEventMsg(contentDiv, 'report', '📊 诊断报告已生成');
                        } else if (type === 'status') {
                            DOM.diagStatusBar.innerHTML = '<span class="diag-dot" style="background:var(--accent-cyan);animation:pulseDot 1.5s infinite"></span> ' + (evt.message || '');
                        } else if (type === 'complete') {
                            var report = (evt.diagnosis && evt.diagnosis.report) || evt.response || '';
                            if (report) contentDiv.innerHTML += marked.parse(report);
                            var doneDiv = document.createElement('div');
                            doneDiv.className = 'ref-source';
                            doneDiv.textContent = '✓ 诊断完成';
                            contentDiv.appendChild(doneDiv);
                            DOM.diagStatusBar.innerHTML = '<span class="diag-dot" style="background:var(--accent-green)"></span> ✓ 诊断完成';
                        } else if (type === 'error') {
                            contentDiv.innerHTML += '<p style="color:var(--accent-red)">❌ ' + escapeHtml(evt.message || '') + '</p>';
                            DOM.diagStatusBar.innerHTML = '<span class="diag-dot" style="background:var(--accent-red)"></span> ❌ 诊断出错';
                        }
                    } catch (e) { /* skip */ }
                }
            }
        }
    } catch (err) {
        contentDiv.innerHTML = '<p style="color:var(--accent-red)">请求失败: ' + escapeHtml(err.message) + '</p>';
    }

    STATE.isStreaming = false;
    DOM.aiopsStartBtn.disabled = false;
    container.scrollTop = container.scrollHeight;
}

// ==================== Session History ====================
async function loadHistory() {
    var DOM = getDom();
    try {
        var resp = await apiGet('/api/chat/sessions');
        var sessions = resp.data || [];
        var list = DOM.chatHistoryList;
        if (!list) return;
        list.innerHTML = '';
        var countEl = document.getElementById('historyCount');
        if (countEl) countEl.textContent = sessions.length;

        sessions.slice(0, 30).forEach(function (s) {
            var div = document.createElement('div');
            div.className = 'history-item';
            var title = s.title || s.session_id || '会话';
            var time = s.updated_at ? s.updated_at.slice(5, 16) : '';
            div.innerHTML = '<span class="history-title">' + escapeHtml(title) + '</span><span class="history-time">' + time + '</span>';
            div.addEventListener('click', function () {
                STATE.currentSessionId = s.session_id;
                loadSessionMessages(s.session_id);
            });
            list.appendChild(div);
        });
    } catch (e) { /* silent */ }
}

async function loadSessionMessages(sessionId) {
    try {
        var resp = await apiGet('/api/chat/session/' + encodeURIComponent(sessionId));
        var history = resp.history || [];
        var container = getDom().chatMessages;
        var welcome = container.querySelector('.welcome-screen');
        if (welcome) welcome.style.display = 'none';
        container.innerHTML = '';
        history.forEach(function (m) {
            if (m.role === 'user' || m.role === 'assistant') {
                addMessage(container, m.role, m.content);
            }
        });
    } catch (e) { /* silent */ }
}

// ==================== Document Management ====================
function initDocUpload() {
    var DOM = getDom();
    if (!DOM.uploadBox) return;

    DOM.uploadBox.addEventListener('click', function () { DOM.fileInput.click(); });

    DOM.fileInput.addEventListener('change', async function (e) {
        var file = e.target.files[0];
        if (!file) return;
        DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-cyan);animation:pulseDot 1.5s infinite"></span> 上传中: ' + file.name + '...';
        try {
            var result = await apiUpload(file);
            if (result.code === 200) {
                DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-green)"></span> ✅ 上传成功: ' + file.name;
            } else {
                DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-red)"></span> ❌ 上传失败: ' + (result.message || '');
            }
        } catch (err) {
            DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-red)"></span> ❌ 上传错误: ' + err.message;
        }
    });

    DOM.indexDocsBtn.addEventListener('click', async function () {
        DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-cyan);animation:pulseDot 1.5s infinite"></span> 索引 uploads 目录...';
        try {
            var result = await apiPost('/api/index_directory', {});
            if (result.code === 200) {
                var d = result.data || {};
                DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-green)"></span> ✅ 索引完成: ' + (d.success_count || 0) + '/' + (d.total_files || 0) + ' 文件';
            } else {
                DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-red)"></span> ❌ 索引失败: ' + (result.message || '');
            }
        } catch (err) {
            DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-red)"></span> ❌ 错误: ' + err.message;
        }
    });

    DOM.indexAiopsBtn.addEventListener('click', async function () {
        DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-cyan);animation:pulseDot 1.5s infinite"></span> 索引 aiops-docs 目录...';
        try {
            var result = await apiPost('/api/index_directory', { directory_path: 'aiops-docs' });
            if (result.code === 200) {
                var d = result.data || {};
                DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-green)"></span> ✅ 索引完成: ' + (d.success_count || 0) + '/' + (d.total_files || 0) + ' 文件';
            } else {
                DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-red)"></span> ❌ 索引失败: ' + (result.message || '');
            }
        } catch (err) {
            DOM.docsStatus.innerHTML = '<span class="status-dot" style="background:var(--accent-red)"></span> ❌ 错误: ' + err.message;
        }
    });
}

// ==================== Eval ====================
async function loadEvalResults() {
    var DOM = getDom();
    try {
        var resp = await apiGet('/api/evaluation/results');
        if (resp.code !== 200 || !resp.data || !resp.data.latest) {
            DOM.evalMetrics.innerHTML = '<div class="eval-placeholder"><svg viewBox="0 0 24 24" fill="none" width="40" height="40"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="1.5" opacity="0.4"/><path d="M12 6v6l4 2" stroke="currentColor" stroke-width="1.5" opacity="0.4"/></svg><p>暂无评估结果</p></div>';
            return;
        }
        var latest = resp.data.latest;
        var metrics = latest.metrics || {};
        var html = '<div class="metrics-grid">';
        var labels = { context_precision: '上下文精确率', context_recall: '上下文召回率', faithfulness: '忠实度', answer_relevancy: '回答相关性' };
        for (var key in labels) {
            var val = metrics[key];
            if (val !== undefined && val !== null) {
                html += '<div class="metric-card"><div class="metric-value">' + (val * 100).toFixed(1) + '%</div><div class="metric-label">' + labels[key] + '</div></div>';
            }
        }
        html += '</div>';
        html += '<div style="margin-top:10px;font-size:12px;color:var(--text-muted);font-family:var(--font-mono)">运行ID: ' + (latest.eval_run_id || '') + ' · 条目: ' + (latest.total_items || 0) + '</div>';
        DOM.evalMetrics.innerHTML = html;

        var details = latest.details || [];
        var dh = '';
        details.slice(0, 5).forEach(function (item, i) {
            dh += '<div class="eval-detail-item"><strong>#' + (i + 1) + '</strong> ' + escapeHtml((item.question || '').slice(0, 120)) + '</div>';
        });
        if (details.length > 5) {
            dh += '<div class="eval-detail-item" style="text-align:center;color:var(--text-muted)">… 还有 ' + (details.length - 5) + ' 条</div>';
        }
        DOM.evalDetails.innerHTML = dh;
    } catch (err) {
        DOM.evalMetrics.innerHTML = '<div class="eval-placeholder"><p>加载失败</p></div>';
    }
}

// ==================== Page Switching ====================
function switchPage(pageName) {
    STATE.currentPage = pageName;
    var DOM = getDom();

    DOM.navBtns.forEach(function (btn) {
        btn.classList.toggle('active', btn.dataset.page === pageName);
    });
    DOM.pages.forEach(function (p) {
        p.classList.toggle('active', p.id === 'page-' + pageName);
        // Re-trigger animation
        if (p.id === 'page-' + pageName) {
            p.style.animation = 'none';
            requestAnimationFrame(function () {
                p.style.animation = '';
            });
        }
    });

    if (pageName === 'eval') loadEvalResults();
    if (pageName === 'chat') loadHistory();
}

// ==================== Init ====================
function init() {
    var DOM = getDom();

    // Navigation
    DOM.navBtns.forEach(function (btn) {
        btn.addEventListener('click', function () { switchPage(btn.dataset.page); });
    });

    // Chat send
    DOM.sendBtn.addEventListener('click', sendMessage);
    DOM.messageInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    DOM.messageInput.addEventListener('input', function () {
        updateCharCount(this, DOM.chatCharCount);
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });

    // RAG send
    DOM.ragSendBtn.addEventListener('click', sendRagQuery);
    DOM.ragInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendRagQuery();
        }
    });
    DOM.ragInput.addEventListener('input', function () {
        updateCharCount(this, DOM.ragCharCount);
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });

    // AIOps
    DOM.aiopsStartBtn.addEventListener('click', startDiagnose);

    // New Chat
    DOM.newChatBtn.addEventListener('click', function () {
        STATE.currentSessionId = 'session_' + Date.now();
        DOM.chatMessages.innerHTML = '';
        DOM.ragMessages.innerHTML = '';
        DOM.aiopsMessages.innerHTML = '';
        // Re-show welcome screens
        ['chat', 'rag', 'aiops'].forEach(function (pid) {
            var page = document.getElementById('page-' + pid);
            if (page) {
                var ws = page.querySelector('.welcome-screen');
                if (ws) ws.style.display = '';
            }
        });
    });

    // Mode selector
    if (DOM.agentModeSelect) {
        DOM.agentModeSelect.addEventListener('change', function (e) {
            STATE.agentMode = e.target.value;
        });
    }

    // Suggestion chips
    document.querySelectorAll('.suggestion-chip').forEach(function (chip) {
        chip.addEventListener('click', function () {
            var text = this.dataset.text;
            var page = this.closest('.page');
            if (page) {
                var pageId = page.id;
                if (pageId === 'page-chat') {
                    DOM.messageInput.value = text;
                    updateCharCount(DOM.messageInput, DOM.chatCharCount);
                    sendMessage();
                } else if (pageId === 'page-rag') {
                    DOM.ragInput.value = text;
                    updateCharCount(DOM.ragInput, DOM.ragCharCount);
                    sendRagQuery();
                }
            }
        });
    });

    // Document management
    initDocUpload();

    // Eval buttons
    DOM.evalGenerateBtn.addEventListener('click', async function () {
        showLoading('生成测试集中...');
        try {
            var resp = await apiPost('/api/evaluation/generate_dataset', { source_dir: 'aiops-docs', count: 10 });
            if (resp.code === 200) {
                showNotification('生成成功: ' + ((resp.data && resp.data.total) || 0) + ' 条', 'success');
            } else {
                showNotification('生成失败: ' + (resp.message || ''), 'error');
            }
        } catch (err) {
            showNotification('错误: ' + err.message, 'error');
        }
        hideLoading();
        loadEvalResults();
    });

    DOM.evalRunBtn.addEventListener('click', async function () {
        showLoading('运行评估中...');
        try {
            var resp = await apiPost('/api/evaluation/run', { use_dataset: true });
            if (resp.code === 200) {
                showNotification('评估完成', 'success');
            } else {
                showNotification('评估失败: ' + (resp.message || ''), 'error');
            }
        } catch (err) {
            showNotification('错误: ' + err.message, 'error');
        }
        hideLoading();
        loadEvalResults();
    });

    DOM.evalRefreshBtn.addEventListener('click', loadEvalResults);

    // Initial load
    loadHistory();
}

document.addEventListener('DOMContentLoaded', init);
