/**
 * Makerspace RAG - Theme Management
 * Light/Dark theme toggle functionality
 */

const Theme = {
    init() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateMermaidTheme(savedTheme === 'dark');
    },

    toggle() {
        const html = document.documentElement;
        const current = html.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        this.updateMermaidTheme(next === 'dark');
    },

    updateMermaidTheme(isDark) {
        if (typeof mermaid !== 'undefined') {
            mermaid.initialize({
                startOnLoad: false,
                theme: isDark ? 'dark' : 'default',
                themeVariables: {
                    primaryColor: '#E5A124',
                    primaryTextColor: isDark ? '#f1f1f1' : '#1a1a2e',
                    primaryBorderColor: '#c98b1d',
                    lineColor: '#E5A124',
                    secondaryColor: '#f5b835',
                    tertiaryColor: isDark ? '#1e1e28' : '#e9ecef',
                    background: isDark ? '#16161d' : '#ffffff',
                    mainBkgColor: isDark ? '#1e1e28' : '#ffffff',
                    textColor: isDark ? '#f1f1f1' : '#1a1a2e'
                },
                darkMode: isDark
            });
        }
    }
};

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', () => Theme.init());
