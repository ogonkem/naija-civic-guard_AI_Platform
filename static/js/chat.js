const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

// Retrieve the CSRF token from the DOM attribute
const csrfToken = document.body.getAttribute('data-csrf');

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
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({ query: text })
        });

        // 3. Consume the newline-delimited JSON stream, rendering tokens live.
        const loadingEl = document.getElementById(loadingId);
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
