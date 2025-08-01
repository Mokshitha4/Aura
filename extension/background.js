// background.js
// This is the service worker for the Chrome Extension.
// It creates the right-click menu and sends data to our FastAPI backend.

// The URL of your locally running FastAPI agent's supervisor endpoint
const AURA_API_URL = "http://127.0.0.1:8000/handle";

// Create a context menu item that appears when text is selected
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "sendToAura",
    title: "Send to Aura",
    contexts: ["selection"]
  });
  console.log("Aura context menu created.");
});

// Listener for when the context menu item is clicked
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "sendToAura" && info.selectionText) {
    const selectedText = info.selectionText;
    console.log(`Sending selected text to Aura: "${selectedText.substring(0, 50)}..."`);

    // Send the data to the backend supervisor
    fetch(AURA_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: selectedText
      }),
    })
    .then(response => {
      if (!response.ok) {
        // If the server response is not OK, read the error message
        return response.json().then(errorData => {
          throw new Error(`Server responded with ${response.status}: ${errorData.detail || 'Unknown error'}`);
        });
      }
      return response.json();
    })
    .then(data => {
      console.log("Successfully sent to Aura. Response:", data.response);
      // Optional: You could show a desktop notification to the user here
      // to confirm the data was saved and maybe show Aura's response.
    })
    .catch(error => {
      console.error("Failed to send to Aura:", error);
      // Optional: Notify the user that the save failed.
    });
  }
});

