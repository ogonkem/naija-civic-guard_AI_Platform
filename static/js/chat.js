const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

// Gateway API key, injected into the page by the chat_page view.
const apiKey = document.querySelector('meta[name="api-key"]')?.content || '';

const esc = (s) => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// One "the agent is doing X" line for a classify/retrieve/chain/verify event.
function agentLine(p) {
    const ms = p.ms != null ? ` · ${p.ms} ms` : '';
    const tools = (p.tools || [])
        .map(([name, tms, ok]) => `${name} ${tms}ms${ok ? '' : ' ✗'}`).join(', ');
    if (p.node === 'classify') return `<b>classify</b> → <code>${esc(p.label)}</code>${ms}`;
    if (p.node === 'retrieve') return `<b>retrieve</b> · ${p.calls} call(s)${ms}`
        + (tools ? ` · ${esc(tools)}` : '')
        + (p.sources?.length ? `<br><span class="dim">↳ ${esc(p.sources.join(', '))}</span>` : '');
    if (p.node === 'chain') return `<b>chain</b>${ms}`
        + (tools ? ` · ${esc(tools)}` : '')
        + (p.chained?.length ? ` · +${esc(p.chained.join(', '))}` : ' · no new cross-refs');
    if (p.node === 'verify') return `<b>verify</b>${ms} · ${p.retry ? '⚠ inadequate → retrying once' : 'ok'}`;
    return `<b>${esc(p.node)}</b>${ms}`;
}

function doneSummary(d) {
    const t = d.timings_ms || {};
    const a = d.agent || {};
    const stage = (k) => t[k] != null ? `${k} ${Math.round(t[k])}ms` : null;
    const stages = ['classify', 'retrieve', 'chain', 'verify', 'generation', 'total']
        .map(stage).filter(Boolean).join(' · ');
    const calls = (d.mcp_tool_calls || [])
        .map(c => `${c.tool_name} ${Math.round(c.tool_latency_ms)}ms${c.ok ? '' : ' ✗'}`).join(' · ') || '—';
    return `<details class="agent-summary"><summary>agent trace`
        + ` — ${esc(a.classify_label || '?')}, ${a.retrieval_calls} retrieval call(s)`
        + `${a.verify_retry ? ', 1 retry' : ''}</summary>`
        + `<div class="dim">nodes: ${esc(stages)}</div>`
        + `<div class="dim">MCP tool calls: ${esc(calls)}</div></details>`;
}

async function handleSend() {
    const text = userInput.value.trim();
    if (!text) return;

    chatWindow.innerHTML += `<div class="msg user">${esc(text)}</div>`;
    userInput.value = "";

    const loadingId = 'loading-' + Date.now();
    chatWindow.innerHTML += `
        <div class="msg bot" id="${loadingId}">
          <div class="agent-trace"></div>
          <div class="answer"><div class="loading-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>
        </div>`;
    chatWindow.scrollTop = chatWindow.scrollHeight;

    const botEl = document.getElementById(loadingId);
    const traceEl = botEl.querySelector('.agent-trace');
    const answerEl = botEl.querySelector('.answer');

    try {
        const response = await fetch('/api/chat/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
            body: JSON.stringify({ query: text })
        });

        if (!response.ok) {
            let detail = '';
            try { detail = (await response.json()).detail || ''; } catch (e) {}
            answerEl.textContent =
                response.status === 429 ? (detail || 'Rate limit reached — wait a moment and try again.') :
                response.status === 401 ? 'This page is not authorized (missing or invalid API key).' :
                `Request failed (HTTP ${response.status}).`;
            traceEl.remove();
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '', answer = '', sources = [], done = null, started = false;

        while (true) {
            const { value, done: streamDone } = await reader.read();
            if (streamDone) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                let p; try { p = JSON.parse(line); } catch (e) { continue; }

                if (p.type === 'agent') {
                    traceEl.insertAdjacentHTML('beforeend', `<div class="agent-step">${agentLine(p)}</div>`);
                } else if (p.type === 'metadata') {
                    sources = p.sources || [];
                } else if (p.type === 'token') {
                    if (!started) { answerEl.textContent = ''; started = true; }
                    answer += p.text || '';
                    answerEl.textContent = answer;
                } else if (p.type === 'done') {
                    done = p;
                } else if (p.type === 'error') {
                    answerEl.textContent = 'Sorry, I encountered an error (' + (p.error || 'unknown') + ').';
                }
                chatWindow.scrollTop = chatWindow.scrollHeight;
            }
        }

        const srcTag = sources.length
            ? `<br><small style="color:#008751"><b>Sources:</b> ${esc(sources.join(', '))}</small>` : '';
        const timeTag = done?.duration
            ? `<br><small class="dim">⏱️ ${done.duration}s total</small>` : '';
        answerEl.innerHTML = `${esc(answer)}${srcTag}${timeTag}`;
        if (done) traceEl.insertAdjacentHTML('beforeend', doneSummary(done));
    } catch (err) {
        answerEl.textContent = "Sorry, I encountered an error. Please try again.";
    }
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleSend(); });
