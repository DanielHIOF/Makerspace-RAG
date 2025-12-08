/**
 * Makerspace RAG - Utility Functions
 * Common helper functions used across the application
 */

const Utils = {
    /**
     * Escape HTML special characters to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    /**
     * Format a message with markdown-like formatting
     * Handles code blocks, links, bold, italic, etc.
     * @param {string} text - Text to format
     * @returns {string} Formatted HTML
     */
    formatMessage(text) {
        // Store Mermaid diagrams for later rendering
        let mermaidDiagrams = [];
        text = text.replace(/```mermaid\n([\s\S]*?)```/g, (match, diagram) => {
            const id = 'mermaid-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
            mermaidDiagrams.push({ id: id, content: diagram.trim() });
            return `<div class="mermaid" id="${id}">${diagram.trim()}</div>`;
        });

        // Handle code blocks
        text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            if (lang === 'mermaid') return match;
            const langClass = lang ? ` class="language-${lang}"` : '';
            return `<pre><code${langClass}>${code.trim()}</code></pre>`;
        });

        // Handle markdown links: [text](url)
        text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener noreferrer" class="chat-link">$1</a>');

        // Handle raw URLs (not already in href)
        text = text.replace(/(?<!href="|">)(https?:\/\/[^\s<]+)/g,
            '<a href="$1" target="_blank" rel="noopener noreferrer" class="chat-link">$1</a>');

        // Handle line breaks
        text = text.replace(/\n/g, '<br>');

        // Handle markdown formatting
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        text = text.replace(/`(.*?)`/g, '<code>$1</code>');

        // Render Mermaid diagrams after DOM is updated
        if (mermaidDiagrams.length > 0) {
            setTimeout(() => {
                mermaidDiagrams.forEach(diagram => {
                    const element = document.getElementById(diagram.id);
                    if (element && typeof mermaid !== 'undefined') {
                        mermaid.run({ nodes: [element] });
                    }
                });
            }, 100);
        }

        return text;
    },

    /**
     * Show a toast notification
     * @param {string} message - Message to show
     * @param {string} type - Type: 'success', 'error', 'warning'
     * @param {number} duration - Duration in ms (default 3000)
     */
    showToast(message, type = 'success', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(-20px)';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },

    /**
     * Debounce a function
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in ms
     * @returns {Function} Debounced function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Format a date for display
     * @param {Date|string} date - Date to format
     * @returns {string} Formatted date string
     */
    formatDate(date) {
        if (typeof date === 'string') {
            date = new Date(date);
        }
        return date.toLocaleDateString('no-NO', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    },

    /**
     * Format file size for display
     * @param {number} bytes - Size in bytes
     * @returns {string} Formatted size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
};
