(() => {
    const API_URL = "https://ai-chatbot-8c3o.onrender.com/chat";
    //const API_URL = "https://localhost:8000/chat";

    const USER_ID = "web_" + Math.random().toString(36).substring(2, 10);

    /* ---------------- UI ---------------- */
    const widget = document.getElementById("chatbot-widget");

    widget.innerHTML = `
        <div id="chat-toggle">
            <div class="toggle-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2Z" fill="currentColor"/>
                </svg>
            </div>
            <div class="toggle-glow"></div>
        </div>

        <div id="chat-box" class="hidden">
            <div id="chat-header">
                <div class="header-content">
                    <div class="header-avatar">
                        <div class="avatar-inner">AI</div>
                        <div class="avatar-ring"></div>
                    </div>
                    <div class="header-text">
                        <div class="header-title">المساعد الذكي</div>
                        <div class="header-status">
                            <span class="status-dot online"></span>
                            <span class="status-text">متصل الآن</span>
                        </div>
                    </div>
                </div>
                <div class="header-actions">
                  
                    <button id="chat-close" class="header-btn" title="إغلاق">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                            <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                    </button>
                </div>
            </div>

           

            <div id="chat-messages"></div>

            <div id="chat-input-area">
                <div class="input-wrapper">
                    <div class="input-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                            <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" 
                                  stroke="#667eea" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <input 
                        id="chat-input" 
                        type="text" 
                        placeholder="اكتب سؤالك هنا..." 
                        autocomplete="off"
                        dir="auto"
                        spellcheck="false"
                    />
                    <div class="input-actions">
                        <button id="chat-send" class="send-btn" title="إرسال">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                                <path d="M22 2L11 13M22 2L15 22L11 13M22 2L2 9L11 13" 
                                      stroke="white" 
                                      stroke-width="2" 
                                      stroke-linecap="round" 
                                      stroke-linejoin="round"/>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="input-footer">
                    <span class="hint-text">اضغط <kbd>Enter</kbd> للإرسال أو <kbd>Esc</kbd> للإلغاء</span>
                </div>
            </div>
        </div>
    `;

    /* ---------------- Updated Styles ---------------- */
    const style = document.createElement("style");
    style.textContent = `
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap');

        body {
            font-family: 'IBM Plex Sans Arabic', Arial, sans-serif;
        }

        /* Toggle Button */
        #chat-toggle {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 64px;
            height: 64px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 9999;
            box-shadow: 
                0 10px 30px rgba(102, 126, 234, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 3px solid white;
        }

        #chat-toggle:hover {
            transform: translateY(-4px) rotate(10deg);
            box-shadow: 
                0 15px 40px rgba(102, 126, 234, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }

        .toggle-icon {
            animation: float 3s ease-in-out infinite;
        }

        .toggle-glow {
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            opacity: 0.2;
            animation: pulse 2s infinite;
            z-index: -1;
        }

        /* Chat Box */
        #chat-box {
            position: fixed;
            bottom: 110px;
            right: 30px;
            width: 400px;
            height: 560px;
            background: white;
            border-radius: 24px;
            box-shadow: 
                0 25px 50px -12px rgba(0, 0, 0, 0.15),
                0 0 0 1px rgba(0, 0, 0, 0.05);
            display: flex;
            flex-direction: column;
            z-index: 9998;
            overflow: hidden;
            opacity: 1;
            visibility: visible;
            transform: translateY(0) scale(1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        #chat-box.hidden {
            opacity: 0;
            visibility: hidden;
            transform: translateY(20px) scale(0.95);
        }

        /* Header */
        #chat-header {
            background: white;
            padding: 20px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #f0f0f0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        }

        .header-content {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .header-avatar {
            position: relative;
            width: 48px;
            height: 48px;
        }

        .avatar-inner {
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 16px;
            z-index: 2;
            position: relative;
        }

        .avatar-ring {
            position: absolute;
            width: calc(100% + 8px);
            height: calc(100% + 8px);
            top: -4px;
            left: -4px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea33, #764ba233);
            z-index: 1;
            animation: spin 4s linear infinite;
        }

        .header-text {
            text-align: right;
            direction: rtl;
        }

        .header-title {
            font-size: 18px;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 4px;
        }

        .header-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #666;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #48bb78;
        }

        .status-dot.online {
            animation: pulse 2s infinite;
            background: #48bb78;
        }

        .header-actions {
            display: flex;
            gap: 8px;
        }

        .header-btn {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            border: none;
            background: #f8f9fa;
            color: #666;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .header-btn:hover {
            background: #f0f0f0;
            color: #333;
            transform: scale(1.1);
        }

        /* Welcome Message */
        #welcome-message {
            padding: 24px;
            border-bottom: 1px solid #f0f0f0;
            background: linear-gradient(to bottom, #fafbff, white);
            transition: all 0.3s ease;
        }

        #welcome-message.hidden {
            display: none;
        }

        .welcome-content {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 20px;
            direction: rtl;
        }

        .welcome-icon {
            width: 56px;
            height: 56px;
            background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
        }

        .welcome-text h3 {
            font-size: 20px;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 4px;
        }

        .welcome-text p {
            font-size: 14px;
            color: #666;
            line-height: 1.5;
        }

        .quick-questions {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .quick-btn {
            padding: 12px 16px;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            color: #333;
            font-family: inherit;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            text-align: right;
            direction: rtl;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .quick-btn:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
            transform: translateX(-4px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }

        /* Messages Container */
        #chat-messages {
            flex: 1;
            padding: 20px 24px;
            overflow-y: auto;
            background: white;
            scrollbar-width: thin;
            scrollbar-color: #c1c1c1 #f0f0f0;
        }

        #chat-messages::-webkit-scrollbar {
            width: 6px;
        }

        #chat-messages::-webkit-scrollbar-track {
            background: #f0f0f0;
            border-radius: 10px;
        }

        #chat-messages::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 10px;
        }

        #chat-messages::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }

        /* Message Bubbles */
        .msg-row {
            display: flex;
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease-out;
        }

        .msg-row.user {
            justify-content: flex-end;
        }

        .msg-row.bot {
            justify-content: flex-start;
        }

        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            flex-shrink: 0;
            margin-top: 4px;
        }

        .avatar.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 12px;
        }

        .avatar.bot {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
            margin-right: 12px;
        }

        .bubble {
            max-width: 75%;
            padding: 16px 20px;
            border-radius: 20px;
            font-size: 15px;
            line-height: 1.6;
            word-wrap: break-word;
            white-space: pre-wrap;
            text-align: right;
            direction: rtl;
            font-feature-settings: "kern", "liga", "clig", "calt";
        }

        .bubble.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }

        .bubble.bot {
            background: #f8f9fa;
            color: #333;
            border-bottom-left-radius: 4px;
            border: 1px solid #f0f0f0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }

        /* Compact Typing Animation - FIXED */
        /* Jump-off typing animation */
        .typing {
            display: flex;
            gap: 6px;
            padding: 1px 0; /* Minimal padding */
            height: 20px; /* Fixed small height */
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: visible; /* Allow dots to go outside */
        }
        
        .typing span {
            width: 5px;
            height: 5px;
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            border-radius: 50%;
            animation: jump 1.4s infinite both;
            position: relative;
        }
        
        .typing span:nth-child(2) { animation-delay: .2s; }
        .typing span:nth-child(3) { animation-delay: .4s; }
        
        @keyframes jump {
            0%, 100% { 
                opacity: 0.3; 
                transform: translateY(0); 
            }
            25% { 
                opacity: 0.7; 
                transform: translateY(-8px); /* Jump high up */
            }
            50% { 
                opacity: 1; 
                transform: translateY(-12px); /* Maximum jump height */
            }
            75% { 
                opacity: 0.7; 
                transform: translateY(-8px); /* Coming down */
            }
        }

        .typing-bubble {
            min-height: auto !important;
            padding: 6px 16px !important; /* Reduced from 8px 16px */
            display: flex;
            align-items: center;
        }

        /* Input Area */
        #chat-input-area {
            padding: 20px 24px;
            border-top: 1px solid #f0f0f0;
            background: white;
        }

        .input-wrapper {
            position: relative;
            display: flex;
            align-items: center;
            background: #f8f9fa;
            border-radius: 16px;
            border: 2px solid #e9ecef;
            transition: all 0.3s;
            overflow: hidden;
        }

        .input-wrapper:focus-within {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            background: white;
        }

        .input-icon {
            padding: 0 16px;
            color: #667eea;
        }

        #chat-input {
            flex: 1;
            padding: 16px 0;
            border: none;
            outline: none;
            font-size: 15px;
            background: transparent;
            color: #333;
            font-family: inherit;
            text-align: right;
            direction: rtl;
        }

        #chat-input::placeholder {
            color: #999;
            text-align: right;
        }

        .send-btn {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            margin: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .send-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        .input-footer {
            margin-top: 12px;
            text-align: center;
        }

        .hint-text {
            font-size: 12px;
            color: #999;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            direction: rtl;
        }

        kbd {
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            border: 1px solid #ddd;
        }

        /* Animations */
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.2; }
            50% { transform: scale(1.2); opacity: 0.3; }
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        @keyframes blink {
            0%, 100% { opacity: 0.2; transform: translateY(0); }
            50% { opacity: 1; transform: translateY(-2px); } /* Reduced movement */
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Responsive Design */
        @media (max-width: 480px) {
            #chat-box {
                width: calc(100vw - 40px);
                right: 20px;
                bottom: 90px;
                height: 500px;
            }
            
            #chat-toggle {
                right: 20px;
                bottom: 20px;
            }
        }

        /* RTL Support */
        [dir="rtl"] .header-text,
        [dir="rtl"] .welcome-content,
        [dir="rtl"] .quick-btn,
        [dir="rtl"] .bubble {
            text-align: right;
        }

        /* Print styles */
        @media print {
            #chat-toggle,
            #chat-box {
                display: none !important;
            }
        }
    `;
    document.head.appendChild(style);

    /* ---------------- JavaScript Logic ---------------- */
    const toggle = document.getElementById("chat-toggle");
    const box = document.getElementById("chat-box");
    const messages = document.getElementById("chat-messages");
    const input = document.getElementById("chat-input");
    const sendBtn = document.getElementById("chat-send");
    const closeBtn = document.getElementById("chat-close");
    //const closeWelcomeBtn = document.getElementById("close-welcome-btn");
    //const welcomeMsg = document.getElementById("welcome-message");
    const quickBtns = document.querySelectorAll(".quick-btn");

    // Toggle chat visibility
    toggle.onclick = () => {
        box.classList.toggle("hidden");
        if (!box.classList.contains("hidden")) {
            input.focus();
        }
    };

    // Close chat
    closeBtn.onclick = () => {
        box.classList.add("hidden");
    };

    // Close welcome message
    // closeWelcomeBtn.onclick = () => {
    //     welcomeMsg.classList.add("hidden");
    //     scrollBottom();
    // };

    // Quick question buttons
    quickBtns.forEach(btn => {
        btn.onclick = () => {
            const msg = btn.getAttribute("data-msg");
            input.value = msg;
            sendMessage();
        };
    });

    // Auto-scroll to bottom
    function scrollBottom() {
        messages.scrollTop = messages.scrollHeight;
    }

    // Add message to chat
    function addMessage(text, sender) {
        const row = document.createElement("div");
        row.className = `msg-row ${sender}`;

        const avatar = document.createElement("div");
        avatar.className = `avatar ${sender}`;
        avatar.textContent = sender === "user" ? "👨‍💼" : "🤖";

        const bubble = document.createElement("div");
        bubble.className = `bubble ${sender}`;
        bubble.textContent = text;

        if (sender === "user") {
            row.appendChild(bubble);
            row.appendChild(avatar);
            // Hide welcome message after first user message
            
        } else {
            row.appendChild(avatar);
            row.appendChild(bubble);
        }

        messages.appendChild(row);
        scrollBottom();
    }

    // Render returned categories with sample products
    function renderCategories(categories) {
        const row = document.createElement("div");
        row.className = "msg-row bot";

        const avatar = document.createElement("div");
        avatar.className = "avatar bot";
        avatar.textContent = "🤖";

        const bubble = document.createElement("div");
        bubble.className = "bubble bot";

        let html = '<div class="categories-list" style="text-align:right;direction:rtl">';
        html += '<strong>الفئات المتوفرة:</strong><br/>';
        categories.forEach(cat => {
            html += `<div style="margin-top:8px"><strong>${cat.category}</strong> — ${cat.count} منتج(s)<br/>`;
            if (Array.isArray(cat.products) && cat.products.length > 0) {
                html += '<ul style="margin:6px 0 0 0; padding-left:18px; text-align:right;">';
                cat.products.slice(0,3).forEach(p => {
                    html += `<li>${p.name} - ${p.price ? p.price + ' USD' : ''} ${p.in_stock ? ' - متوفر' : ' - غير متوفر'}</li>`;
                });
                html += '</ul>';
            }
            html += '</div>';
        });
        html += '</div>';

        bubble.innerHTML = html;
        row.appendChild(avatar);
        row.appendChild(bubble);
        messages.appendChild(row);
        scrollBottom();
    }

    // Show typing indicator - COMPACT VERSION
    function showTyping() {
        const row = document.createElement("div");
        row.className = "msg-row bot";
        row.id = "typing-indicator";

        row.innerHTML = `
            <div class="avatar bot">🤖</div>
            <div class="bubble bot typing-bubble">
                <div class="typing">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;

        messages.appendChild(row);
        scrollBottom();
    }

    // Remove typing indicator
    function removeTyping() {
        const typing = document.getElementById("typing-indicator");
        if (typing) typing.remove();
    }

    // Send message to API
    async function sendMessage() {
        const text = input.value.trim();
        if (!text) return;

        addMessage(text, "user");
        input.value = "";

        showTyping();

        try {
            const res = await fetch(API_URL, {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({
                    user_id: USER_ID,
                    message: text,
                    timestamp: new Date().toISOString()
                })
            });

            if (!res.ok) {
                throw new Error(`HTTP error! status: ${res.status}`);
            }

            const data = await res.json();
            removeTyping();
            
            if (data.response) {
                addMessage(data.response, "bot");
                // If backend returned grouped categories, render them visually
                if (data.data && Array.isArray(data.data.categories) && data.data.categories.length > 0) {
                    renderCategories(data.data.categories);
                }
            } else {
                addMessage("عذراً، لم أتمكن من معالجة طلبك. الرجاء المحاولة مرة أخرى.", "bot");
            }

        } catch (error) {
            console.error("Chat error:", error);
            removeTyping();
            addMessage("تعذر الاتصال بالخادم. يرجى التحقق من اتصال الإنترنت والمحاولة مرة أخرى.", "bot");
        }
    }

    // Event listeners
    sendBtn.onclick = sendMessage;
    
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        } else if (e.key === "Escape") {
            input.value = "";
            input.blur();
        }
    });

    // Auto-focus input when chat opens
    input.addEventListener("focus", () => {
        box.classList.remove("hidden");
    });

    // Welcome message when opening chat
    if (!box.classList.contains("hidden")) {
        setTimeout(() => {
            input.focus();
        }, 300);
    }

    // Close chat when clicking outside
    document.addEventListener("click", (e) => {
        if (!box.classList.contains("hidden") && 
            !box.contains(e.target) && 
            !toggle.contains(e.target)) {
            box.classList.add("hidden");
        }
    });

    // Initialize
    console.log("🤖 Chatbot widget initialized successfully!");
})();