/**
 * Makerspace RAG - API Client
 * Centralized API calls for the application
 */

const API = {
    /**
     * Send a chat message
     * @param {string} message - The user's message
     * @param {Array} history - Conversation history
     * @param {string} summary - Compressed conversation summary
     * @returns {Promise<Object>} Response with response text and optional summary
     */
    async chat(message, history = [], summary = '') {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, history, summary })
        });
        return res.json();
    },

    /**
     * Get system status
     * @returns {Promise<Object>} System status info
     */
    async getStatus() {
        const res = await fetch('/status');
        return res.json();
    },

    /**
     * Get health check
     * @returns {Promise<Object>} Health check result
     */
    async getHealth() {
        const res = await fetch('/health');
        return res.json();
    },

    /**
     * Reload the knowledge index
     * @returns {Promise<Object>} Reload result
     */
    async reloadIndex() {
        const res = await fetch('/reload', {
            method: 'POST'
        });
        return res.json();
    },

    /**
     * Add text to the knowledge base
     * @param {string} text - Text to add
     * @returns {Promise<Object>} Result
     */
    async addText(text) {
        const res = await fetch('/add-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        return res.json();
    },

    // Component API methods
    components: {
        /**
         * Get all components or search
         * @param {Object} options - Search options
         * @returns {Promise<Array>} List of components
         */
        async list(options = {}) {
            const params = new URLSearchParams();
            if (options.query) params.set('q', options.query);
            if (options.restockOnly) params.set('restock', 'true');
            const url = '/api/components' + (params.toString() ? '?' + params : '');
            const res = await fetch(url);
            return res.json();
        },

        /**
         * Get a single component
         * @param {number} id - Component ID
         * @returns {Promise<Object>} Component data
         */
        async get(id) {
            const res = await fetch(`/api/components/${id}`);
            return res.json();
        },

        /**
         * Create a new component
         * @param {Object} data - Component data
         * @returns {Promise<Object>} Created component
         */
        async create(data) {
            const res = await fetch('/api/components', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            return res.json();
        },

        /**
         * Update a component
         * @param {number} id - Component ID
         * @param {Object} data - Updated data
         * @returns {Promise<Object>} Updated component
         */
        async update(id, data) {
            const res = await fetch(`/api/components/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            return res.json();
        },

        /**
         * Delete a component
         * @param {number} id - Component ID
         * @returns {Promise<Object>} Result
         */
        async delete(id) {
            const res = await fetch(`/api/components/${id}`, {
                method: 'DELETE'
            });
            return res.json();
        },

        /**
         * Get all shelf locations
         * @returns {Promise<Array>} List of locations
         */
        async getLocations() {
            const res = await fetch('/api/hylleplasser');
            return res.json();
        }
    }
};
