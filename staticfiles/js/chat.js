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
        
        const data = await response.json();

        // 3. Replace Loading with Actual Answer
        const loadingEl = document.getElementById(loadingId);
        const sourceTag = data.sources ? `<br><small style="color:#008751"><b>Sources:</b> ${data.sources.join(', ')}</small>` : '';
        const timeTag = data.duration ? `<br><small style="color:#888; font-size: 11px;">⏱️ Generation time: ${data.duration}s</small>` : '';
        
        loadingEl.innerHTML = `${data.answer} ${sourceTag} ${timeTag}`;
    } catch (err) {
        document.getElementById(loadingId).innerHTML = "Sorry, I encountered an error. Please try again.";
    }
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSend();
});
