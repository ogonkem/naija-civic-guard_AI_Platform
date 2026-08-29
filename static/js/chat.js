const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

// Gateway API key, injected into the page by the chat_page view.
const apiKey = document.querySelector('meta[name="api-key"]')?.content || '';

async function handleSend() {
    const text = userInput.value.trim();
    if (!text) return;

    // 1. Add User Message
    chatWindow.innerHTML += `<div class="msg user">${text}</div>`;
    userInput.value = "";

    // 2. Add Loading Message
    const loadingId = 'loading-' + Date.now();
    chatWindow.innerHTML += `
        <div class="msg bot" id="${loadingId}">
            <div class="loading-dots">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>
        </div>`;
    chatWindow.scrollTop = chatWindow.scrollHeight;

    try {
        const response = await fetch('/api/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKey
            },
            body: JSON.stringify({ query: text })
        });

        const loadingEl = document.getElementById(loadingId);

        // Gateway rejections (401 / 429) come back as a JSON error, not a stream.
        if (!response.ok) {
            let detail = '';
            try { detail = (await response.json()).detail || ''; } catch (e) {}
            loadingEl.textContent =
                response.status === 429 ? (detail || 'Rate limit reached — wait a moment and try again.') :
                response.status === 401 ? 'This page is not authorized (missing or invalid API key).' :
                `Request failed (HTTP ${response.status}).`;
            chatWindow.scrollTop = chatWindow.scrollHeight;
            return;
        }

        // 3. Consume the newline-delimited JSON stream, rendering tokens live.
        loadingEl.textContent = '';

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let answer = '';
        let sources = [];
        let duration = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop();  // keep the trailing partial line

            for (const line of lines) {
                if (!line.trim()) continue;
                let payload;
                try { payload = JSON.parse(line); } catch (e) { continue; }

                if (payload.type === 'metadata') {
                    sources = payload.sources || [];
                } else if (payload.type === 'token') {
                    answer += payload.text || '';
                    loadingEl.textContent = answer;
                    chatWindow.scrollTop = chatWindow.scrollHeight;
                } else if (payload.type === 'done') {
                    duration = payload.duration;
                } else if (payload.type === 'error') {
                    answer = "Sorry, I encountered an error. Please try again.";
                    loadingEl.textContent = answer;
                }
            }
        }

        const sourceTag = sources.length
            ? `<br><small style="color:#008751"><b>Sources:</b> ${sources.join(', ')}</small>`
            : '';
        const timeTag = duration
            ? `<br><small style="color:#888; font-size: 11px;">⏱️ Generation time: ${duration}s</small>`
            : '';
        loadingEl.innerHTML = `${answer}${sourceTag}${timeTag}`;
    } catch (err) {
        document.getElementById(loadingId).innerHTML = "Sorry, I encountered an error. Please try again.";
    }
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSend();
});
