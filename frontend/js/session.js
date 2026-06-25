const EmotionSession = (() => {
    const API_BASE = window.EMOTION_API_BASE || 'http://127.0.0.1:8000';
    const USER_KEY = 'emotionCultureUser';
    const LAST_RUN_KEY = 'emotionCultureLastRun';
    const DEFAULT_USER = { user_id: 1, username: 'demo', display_name: 'Demo User' };

    function currentUser() {
        try {
            return JSON.parse(localStorage.getItem(USER_KEY) || '') || DEFAULT_USER;
        } catch {
            return DEFAULT_USER;
        }
    }

    function currentUserId() {
        return Number(currentUser().user_id || 1);
    }

    function setUser(user) {
        const next = {
            user_id: Number(user.user_id || 1),
            username: user.username || 'demo',
            display_name: user.display_name || user.username || 'Demo User'
        };
        localStorage.setItem(USER_KEY, JSON.stringify(next));
        updateUserLabels(next);
        return next;
    }

    async function requestJson(path, options = {}) {
        const response = await fetch(`${API_BASE}${path}`, {
            ...options,
            headers: {
                ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
                ...(options.headers || {})
            }
        });
        if (!response.ok) {
            let detail = `${response.status} ${response.statusText}`;
            try {
                const body = await response.json();
                detail = body.detail || detail;
            } catch {
                // Keep the HTTP status text.
            }
            throw new Error(detail);
        }
        return response.json();
    }

    async function login(username, displayName = '') {
        const response = await requestJson('/api/users', {
            method: 'POST',
            body: JSON.stringify({
                username: String(username || '').trim() || 'demo',
                display_name: String(displayName || '').trim()
            })
        });
        return setUser(response.user);
    }

    async function ensureUser() {
        const user = currentUser();
        try {
            const response = await requestJson(`/api/users/${user.user_id}`);
            return setUser(response.user);
        } catch {
            return login(user.username || 'demo', user.display_name || '');
        }
    }

    function updateUserLabels(user = currentUser()) {
        document.querySelectorAll('[data-current-user]').forEach((node) => {
            node.textContent = user.display_name || user.username || 'Demo User';
        });
        document.querySelectorAll('[data-user-input]').forEach((node) => {
            if (!node.value) node.value = user.username || 'demo';
        });
    }

    function bindUserPanel(onChange) {
        updateUserLabels();
        document.querySelectorAll('[data-user-login]').forEach((button) => {
            button.addEventListener('click', async () => {
                const panel = button.closest('[data-user-panel]') || document;
                const input = panel.querySelector('[data-user-input]');
                const username = input?.value?.trim() || 'demo';
                button.disabled = true;
                try {
                    const user = await login(username, username);
                    if (typeof onChange === 'function') onChange(user);
                } finally {
                    button.disabled = false;
                }
            });
        });
    }

    function saveLastRun(payload) {
        localStorage.setItem(LAST_RUN_KEY, JSON.stringify({
            ...payload,
            user_id: currentUserId(),
            saved_at: new Date().toISOString()
        }));
    }

    function loadLastRun() {
        try {
            const payload = JSON.parse(localStorage.getItem(LAST_RUN_KEY) || '');
            return payload && Number(payload.user_id) === currentUserId() ? payload : null;
        } catch {
            return null;
        }
    }

    return {
        API_BASE,
        currentUser,
        currentUserId,
        setUser,
        requestJson,
        login,
        ensureUser,
        bindUserPanel,
        saveLastRun,
        loadLastRun
    };
})();

window.EmotionSession = EmotionSession;
