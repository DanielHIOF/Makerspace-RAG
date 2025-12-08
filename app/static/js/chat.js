/**
 * Makerspace RAG - Chat Interface
 * Main chat functionality
 */

const Chat = {
    // State
    selectedLevel: '',
    selectedLang: '',
    conversationHistory: [],
    conversationSummary: '',
    commandSuggestionsVisible: false,
    selectedCommandIndex: -1,

    // Available commands
    commands: [
        { name: '/nybegynner', desc: 'Enkel forklaring for nybegynnere', value: '/nybegynner ' },
        { name: '/ekspert', desc: 'Teknisk og detaljert forklaring', value: '/ekspert ' },
        { name: '/norsk', desc: 'Svar på norsk', value: '/norsk ' },
        { name: '/english', desc: 'Answer in English', value: '/english ' },
        { name: '/prusa', desc: 'Fokus på Prusa-printere og PrusaSlicer', value: '/prusa ' },
        { name: '/3d', desc: 'Fokus på 3D-printing generelt', value: '/3d ' },
        { name: '/laser', desc: 'Fokus på laserkutting', value: '/laser ' },
        { name: '/cnc', desc: 'Fokus på CNC-fresing', value: '/cnc ' },
        { name: '/elektronikk', desc: 'Fokus på elektronikk og Arduino', value: '/elektronikk ' },
        { name: '/lodding', desc: 'Fokus på lodding', value: '/lodding ' }
    ],

    levelDescriptions: {
        '/nybegynner ': 'Nybegynner: Enkelt sprak, steg-for-steg, ingen faguttrykk',
        '': 'Normal: Balansert forklaring med korte definisjoner av faguttrykk',
        '/ekspert ': 'Ekspert: Teknisk presisjon, fagterminologi, avanserte detaljer'
    },

    // DOM Elements
    elements: {},

    init() {
        // Cache DOM elements
        this.elements = {
            chatArea: document.getElementById('chatArea'),
            messages: document.getElementById('messages'),
            welcome: document.getElementById('welcome'),
            input: document.getElementById('input'),
            sendBtn: document.getElementById('sendBtn'),
            levelDescBar: document.getElementById('levelDescBar'),
            commandSuggestions: document.getElementById('commandSuggestions'),
            newChatBtn: document.getElementById('newChatBtn'),
            themeToggle: document.getElementById('themeToggle')
        };

        this.setupEventListeners();
        this.elements.input.focus();
    },

    setupEventListeners() {
        // Level buttons
        document.querySelectorAll('.ctrl-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.ctrl-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.selectedLevel = btn.dataset.value;
                this.elements.levelDescBar.textContent = this.levelDescriptions[this.selectedLevel] || '';
            });
        });

        // Language buttons
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.selectedLang = btn.dataset.value;
            });
        });

        // Input events
        this.elements.input.addEventListener('input', (e) => this.handleInput(e));
        this.elements.input.addEventListener('keydown', (e) => this.handleKeyDown(e));

        // Send button
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());

        // New chat button
        this.elements.newChatBtn.addEventListener('click', () => this.newChat());

        // Theme toggle
        this.elements.themeToggle.addEventListener('click', () => Theme.toggle());

        // Hide suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.elements.input.contains(e.target) &&
                !this.elements.commandSuggestions.contains(e.target)) {
                this.hideCommandSuggestions();
            }
        });
    },

    addMessage(content, isUser, addToHistory = true) {
        this.elements.welcome.style.display = 'none';

        // Create wrapper for message + actions
        const wrapper = document.createElement('div');
        wrapper.className = `message-wrapper ${isUser ? 'user' : 'assistant'}`;

        const div = document.createElement('div');
        div.className = `message ${isUser ? 'user' : 'assistant'}`;
        div.innerHTML = Utils.formatMessage(content);

        // Create action buttons
        const actions = document.createElement('div');
        actions.className = 'message-actions';

        const copyBtn = document.createElement('button');
        copyBtn.className = 'msg-action-btn';
        copyBtn.innerHTML = '&#128203; Kopier';
        copyBtn.title = 'Kopier melding';
        copyBtn.addEventListener('click', () => this.copyMessage(content, copyBtn));
        actions.appendChild(copyBtn);

        wrapper.appendChild(div);
        wrapper.appendChild(actions);
        this.elements.messages.appendChild(wrapper);
        this.elements.chatArea.scrollTop = this.elements.chatArea.scrollHeight;

        if (addToHistory) {
            this.conversationHistory.push({
                role: isUser ? 'user' : 'assistant',
                content: content
            });
            // Keep last 60 messages (30 exchanges)
            if (this.conversationHistory.length > 60) {
                this.conversationHistory = this.conversationHistory.slice(-60);
            }
        }

        return div;
    },

    async copyMessage(content, btn) {
        try {
            await navigator.clipboard.writeText(content);
            btn.classList.add('copied');
            btn.innerHTML = '&#10003; Kopiert!';
            setTimeout(() => {
                btn.classList.remove('copied');
                btn.innerHTML = '&#128203; Kopier';
            }, 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    },

    showLoading() {
        const div = document.createElement('div');
        div.className = 'message assistant';
        div.id = 'loading';
        div.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
        this.elements.messages.appendChild(div);
        this.elements.chatArea.scrollTop = this.elements.chatArea.scrollHeight;
    },

    hideLoading() {
        const loading = document.getElementById('loading');
        if (loading) loading.remove();
    },

    async sendMessage() {
        const text = this.elements.input.value.trim();
        if (!text) return;

        const fullMessage = this.selectedLang + this.selectedLevel + text;
        this.addMessage(text, true);
        this.elements.input.value = '';
        this.elements.sendBtn.disabled = true;
        this.showLoading();

        try {
            const data = await API.chat(
                fullMessage,
                this.conversationHistory.slice(0, -1),
                this.conversationSummary
            );
            this.hideLoading();

            if (data.summary) {
                this.conversationSummary = data.summary;
            }

            this.addMessage(data.response, false);
        } catch (e) {
            this.hideLoading();
            this.addMessage('Beklager, noe gikk galt. Prov igjen.', false, false);
        }

        this.elements.sendBtn.disabled = false;
        this.elements.input.focus();
    },

    askQuestion(q) {
        this.elements.input.value = q;
        this.sendMessage();
    },

    newChat() {
        this.conversationHistory = [];
        this.conversationSummary = '';
        this.elements.messages.innerHTML = '';
        this.elements.welcome.style.display = 'block';
        this.hideCommandSuggestions();
        this.elements.input.focus();
    },

    // Command suggestions
    handleInput(e) {
        const value = e.target.value;
        const lastChar = value[value.length - 1];
        const lastSlashIndex = value.lastIndexOf('/');

        if (lastChar === '/' || (lastSlashIndex !== -1 && value.length > lastSlashIndex)) {
            this.showCommandSuggestions();
        } else {
            this.hideCommandSuggestions();
        }
    },

    handleKeyDown(e) {
        // Ctrl+Enter or Cmd+Enter always sends
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            this.sendMessage();
            return;
        }

        if (!this.commandSuggestionsVisible) {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
            return;
        }

        const items = this.elements.commandSuggestions.querySelectorAll('.command-item');

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            this.selectedCommandIndex = Math.min(this.selectedCommandIndex + 1, items.length - 1);
            this.updateCommandSuggestions();
            items[this.selectedCommandIndex]?.scrollIntoView({ block: 'nearest' });
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            this.selectedCommandIndex = Math.max(this.selectedCommandIndex - 1, -1);
            this.updateCommandSuggestions();
            if (this.selectedCommandIndex >= 0) {
                items[this.selectedCommandIndex]?.scrollIntoView({ block: 'nearest' });
            }
        } else if (e.key === 'Enter' && this.selectedCommandIndex >= 0) {
            e.preventDefault();
            const query = this.elements.input.value.substring(
                this.elements.input.value.lastIndexOf('/') + 1
            ).toLowerCase();
            const filtered = this.commands.filter(c => c.name.toLowerCase().includes(query));
            const cmd = filtered[this.selectedCommandIndex];
            if (cmd) {
                this.insertCommand(cmd.value);
            }
        } else if (e.key === 'Escape') {
            this.hideCommandSuggestions();
        } else if (e.key === 'Enter') {
            this.sendMessage();
        }
    },

    showCommandSuggestions() {
        this.commandSuggestionsVisible = true;
        this.selectedCommandIndex = -1;
        this.updateCommandSuggestions();
        this.elements.commandSuggestions.style.display = 'block';
    },

    hideCommandSuggestions() {
        this.commandSuggestionsVisible = false;
        this.elements.commandSuggestions.style.display = 'none';
    },

    updateCommandSuggestions() {
        const inputValue = this.elements.input.value;
        const query = inputValue.substring(inputValue.lastIndexOf('/') + 1).toLowerCase();

        const filtered = this.commands.filter(cmd =>
            cmd.name.toLowerCase().includes(query)
        );

        this.elements.commandSuggestions.innerHTML = '';

        if (filtered.length === 0) {
            this.hideCommandSuggestions();
            return;
        }

        filtered.forEach((cmd, index) => {
            const item = document.createElement('div');
            item.className = 'command-item' + (index === this.selectedCommandIndex ? ' selected' : '');
            item.innerHTML = `
                <div class="command-name">${cmd.name}</div>
                <div class="command-desc">${cmd.desc}</div>
            `;
            item.addEventListener('click', () => this.insertCommand(cmd.value));
            this.elements.commandSuggestions.appendChild(item);
        });
    },

    insertCommand(commandValue) {
        const inputValue = this.elements.input.value;
        const lastSlashIndex = inputValue.lastIndexOf('/');
        if (lastSlashIndex !== -1) {
            this.elements.input.value = inputValue.substring(0, lastSlashIndex) + commandValue;
        } else {
            this.elements.input.value = commandValue;
        }
        this.hideCommandSuggestions();
        this.elements.input.focus();
    }
};

// Initialize chat when DOM is ready
document.addEventListener('DOMContentLoaded', () => Chat.init());

// Global function for category buttons
function askQuestion(q) {
    Chat.askQuestion(q);
}
