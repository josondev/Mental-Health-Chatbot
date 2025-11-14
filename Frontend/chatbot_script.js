// Configuration - UPDATE THIS AFTER DEPLOYMENT!
const API_BASE_URL = 'http://localhost:8000'; // Change to your Render URL
let chatHistory = [];

// Check API health on load
window.addEventListener('load', checkAPIHealth);

async function checkAPIHealth() {
    const statusElement = document.getElementById('statusText');
    const statusDot = document.querySelector('.status-dot');

    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (response.ok && data.chatbot_ready) {
            statusElement.textContent = 'Online';
            statusDot.style.background = '#4caf50';
        } else {
            statusElement.textContent = 'Limited';
            statusDot.style.background = '#ff9800';
        }
    } catch (error) {
        statusElement.textContent = 'Offline';
        statusDot.style.background = '#f44336';
        console.error('Health check failed:', error);
    }
}

// Send message
async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();

    if (!message) return;

    // Add user message
    addMessage(message, 'user');
    input.value = '';
    input.style.height = 'auto';

    // Show typing
    showTypingIndicator();

    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                chat_history: chatHistory
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        removeTypingIndicator();
        addMessage(data.response, 'bot');
        chatHistory = data.chat_history;

    } catch (error) {
        removeTypingIndicator();
        addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        console.error('Error:', error);
    } finally {
        sendButton.disabled = false;
    }
}

function addMessage(text, role) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    const avatar = role === 'bot' ? '🧠' : '👤';

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-text">${text}</div>
        </div>
    `;

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showTypingIndicator() {
    const messagesDiv = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.className = 'message bot-message';
    typingDiv.innerHTML = `
        <div class="message-avatar">🧠</div>
        <div class="message-content">
            <div class="message-text typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    messagesDiv.appendChild(typingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Auto-resize textarea
const textarea = document.getElementById('userInput');
textarea.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});

// Enter to send
textarea.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
