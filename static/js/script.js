document.addEventListener("DOMContentLoaded", function () {
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatBox = document.getElementById("chat-box");

    chatForm.addEventListener("submit", function (event) {
        event.preventDefault();

        const message = chatInput.value;
        if (!message) return;

        // Add user's message to the chat box
        appendMessage("user", message);

        // Clear the input field
        chatInput.value = "";

        // Call the chat API with the user's message
        fetch("/chat_api", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                stage: 1  // You can use this 'stage' for different parts of the conversation
            })
        })
        .then(response => response.json())
        .then(data => {
            // Append bot's response to the chat box
            appendMessage("bot", data.reply);
        })
        .catch(error => {
            console.error("Error:", error);
        });
    });

    // Function to append a message to the chat box
    function appendMessage(sender, message) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);
        messageDiv.textContent = message;
        chatBox.appendChild(messageDiv);

        // Scroll to the bottom of the chat box
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
