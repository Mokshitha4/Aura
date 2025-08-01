// popup.js
// This script handles the logic for the chat interface in the extension's popup.
// It now includes logic to save and load chat history and uses the correct CSS classes.

const chatForm = document.getElementById('chat-form');
const messageInput = document.getElementById('message-input');
const chatContainer = document.getElementById('chat-container');
const submitButton = document.getElementById('submit-button');
const contextCheckbox = document.getElementById('context-checkbox');

const AURA_API_URL = 'http://127.0.0.1:8000/handle';

// --- Chat History Management ---
let conversationHistory = [];

// Function to save the entire conversation to Chrome's local storage
function saveHistory() {
    chrome.storage.local.set({ 'aura_chat_history': conversationHistory });
}

// Function to load and render history when the popup opens
function loadHistory() {
    chrome.storage.local.get(['aura_chat_history'], (result) => {
        if (result.aura_chat_history && result.aura_chat_history.length > 0) {
            conversationHistory = result.aura_chat_history;
            // Clear the initial welcome message before rendering history
            chatContainer.innerHTML = ''; 
            conversationHistory.forEach(message => {
                addMessage(message.sender, message.text, false); // Don't save again
            });
        }
    });
}

// --- MODIFIED: addMessage function to use manual CSS classes ---
function addMessage(sender, message, save = true) {
    const messageElement = document.createElement('div');
    
    if (save) {
        conversationHistory.push({ sender, text: message });
        saveHistory();
    }
    
    if (sender === 'user') {
        messageElement.className = 'message-wrapper user-message';
        messageElement.innerHTML = `
            <div class="message-bubble">
                <p>${message}</p>
            </div>
            <div class="message-avatar">You</div>
        `;
    } else { // sender === 'aura' or 'loading'
        messageElement.className = 'message-wrapper aura-message';
        if (sender === 'loading') {
            messageElement.id = 'loading-indicator';
            messageElement.innerHTML = `
                <div class="message-avatar">A</div>
                <div class="message-bubble">
                    <div class="loading-dots">
                        <div class="dot-1"></div>
                        <div class="dot-2"></div>
                        <div class="dot-3"></div>
                    </div>
                </div>
            `;
        } else { // sender === 'aura'
            messageElement.innerHTML = `
                <div class="message-avatar">A</div>
                <div class="message-bubble">
                    <p>${message}</p>
                </div>
            `;
        }
    }
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Function to get the content of the current webpage
async function getPageContent() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab) {
        try {
            const results = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: () => document.body.innerText,
            });
            return results[0].result;
        } catch (e) {
            console.error("Could not get page content:", e);
            return null;
        }
    }
    return null;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const userMessage = messageInput.value.trim();
    if (!userMessage) return;

    addMessage('user', userMessage);
    messageInput.value = '';
    
    submitButton.disabled = true;
    messageInput.disabled = true;
    addMessage('loading', '', false); // Don't save loading indicator to history

    let fullPrompt = userMessage;

    if (contextCheckbox.checked) {
        const pageContent = await getPageContent();
        if (pageContent) {
            fullPrompt = `Based on the following webpage content, please answer my question.\n\n--- WEBPAGE CONTENT ---\n${pageContent.substring(0, 4000)}\n\n--- MY QUESTION ---\n${userMessage}`;
        }
    }

    try {
        const response = await fetch(AURA_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: fullPrompt })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'An unknown error occurred.');
        }

        const data = await response.json();
        
        document.getElementById('loading-indicator')?.remove();
        addMessage('aura', data.response);

    } catch (error) {
        console.error('Error communicating with Aura API:', error);
        document.getElementById('loading-indicator')?.remove();
        addMessage('aura', `Sorry, an error occurred: ${error.message}`);
    } finally {
        submitButton.disabled = false;
        messageInput.disabled = false;
        messageInput.focus();
    }
});

// Auto-resize the textarea
messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = (messageInput.scrollHeight) + 'px';
});

// Load history when the popup is opened
document.addEventListener('DOMContentLoaded', loadHistory);
