const EmotionSession = (() => {
    const API_BASE = window.EMOTION_API_BASE || 'http://127.0.0.1:8000';
    const USER_KEY = 'emotionCultureUser';
    const LAST_RUN_KEY = 'emotionCultureLastRun';
    const LOGIN_PAGE = 'login.html';

    function currentUser() {
        try {
            const raw = localStorage.getItem(USER_KEY);
            if (!raw) return null;
            const user = JSON.parse(raw);
            return user && user.user_id ? user : null;
        } catch {
            return null;
        }
    }

    function currentUserId() {
        const user = currentUser();
        return user ? Number(user.user_id) : null;
    }

    function isAuthenticated() {
        return Boolean(currentUser());
    }

    function setUser(user) {
        const next = {
            user_id: Number(user.user_id),
            username: user.username || '',
            display_name: user.display_name || user.username || '',
            is_guest: Boolean(user.is_guest)
        };
        localStorage.setItem(USER_KEY, JSON.stringify(next));
        updateUserLabels(next);
        return next;
    }

    function clearUser() {
        localStorage.removeItem(USER_KEY);
        try {
            localStorage.removeItem(LAST_RUN_KEY);
        } catch {
            // ignore
        }
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

    async function register({ username, password, passwordConfirm, name }) {
        const response = await requestJson('/api/auth/register', {
            method: 'POST',
            body: JSON.stringify({
                username: String(username || '').trim(),
                password: String(password || ''),
                password_confirm: String(passwordConfirm || ''),
                name: String(name || '').trim()
            })
        });
        return setUser(response.user);
    }

    async function login(username, password) {
        const response = await requestJson('/api/auth/login', {
            method: 'POST',
            body: JSON.stringify({
                username: String(username || '').trim(),
                password: String(password || '')
            })
        });
        return setUser(response.user);
    }

    async function guestLogin() {
        const response = await requestJson('/api/auth/guest', {
            method: 'POST',
            body: '{}'
        });
        return setUser(response.user);
    }

    async function checkUsername(username) {
        const response = await requestJson(
            `/api/auth/check-username?username=${encodeURIComponent(String(username || '').trim())}`
        );
        return Boolean(response.available);
    }

    function redirectToLogin() {
        if (!window.location.pathname.endsWith(LOGIN_PAGE)) {
            window.location.href = LOGIN_PAGE;
        }
    }

    function logout() {
        clearUser();
        redirectToLogin();
    }

    function requireAuth() {
        if (!isAuthenticated()) {
            redirectToLogin();
            return false;
        }
        return true;
    }

    async function ensureUser() {
        const user = currentUser();
        if (!user) {
            redirectToLogin();
            return null;
        }
        updateUserLabels(user);
        try {
            const response = await requestJson(`/api/users/${user.user_id}`);
            return setUser(response.user);
        } catch {
            // 서버 일시 오류 등에는 캐시된 세션을 유지한다 (데모로 되돌아가지 않음).
            return user;
        }
    }

    function updateUserLabels(user = currentUser()) {
        const name = user ? (user.display_name || user.username || '') : '';
        document.querySelectorAll('[data-current-user]').forEach((node) => {
            node.textContent = name || '게스트';
        });
    }

    function bindUserPanel() {
        updateUserLabels();
        document.querySelectorAll('[data-logout]').forEach((button) => {
            if (button.dataset.logoutBound) return;
            button.dataset.logoutBound = '1';
            button.addEventListener('click', (event) => {
                event.preventDefault();
                logout();
            });
        });
    }

    function saveLastRun(payload) {
        const userId = currentUserId();
        if (!userId) return;
        localStorage.setItem(LAST_RUN_KEY, JSON.stringify({
            ...payload,
            user_id: userId,
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
        isAuthenticated,
        setUser,
        clearUser,
        requestJson,
        register,
        login,
        guestLogin,
        checkUsername,
        logout,
        requireAuth,
        ensureUser,
        bindUserPanel,
        updateUserLabels,
        saveLastRun,
        loadLastRun
    };
})();

window.EmotionSession = EmotionSession;
