(function () {
    const HOME_PAGE = 'index.html';

    function el(id) {
        return document.getElementById(id);
    }

    function setMessage(node, text, isError = true) {
        if (!node) return;
        node.textContent = text || '';
        node.classList.toggle('auth-message-error', Boolean(text) && isError);
        node.classList.toggle('auth-message-ok', Boolean(text) && !isError);
    }

    function goHome() {
        window.location.href = HOME_PAGE;
    }

    function switchTab(tab) {
        document.querySelectorAll('.auth-tab').forEach((button) => {
            button.classList.toggle('auth-tab-active', button.dataset.tab === tab);
        });
        const loginForm = el('loginForm');
        const registerForm = el('registerForm');
        loginForm.hidden = tab !== 'login';
        registerForm.hidden = tab !== 'register';
        setMessage(el('loginMessage'), '');
        setMessage(el('registerMessage'), '');
    }

    async function handleLogin(event) {
        event.preventDefault();
        const username = el('loginUsername').value.trim();
        const password = el('loginPassword').value;
        const submit = el('loginSubmit');
        if (!username || !password) {
            setMessage(el('loginMessage'), '아이디와 비밀번호를 입력해 주세요.');
            return;
        }
        submit.disabled = true;
        submit.textContent = '로그인 중...';
        try {
            await window.EmotionSession.login(username, password);
            goHome();
        } catch (error) {
            setMessage(el('loginMessage'), error.message || '로그인에 실패했습니다.');
            submit.disabled = false;
            submit.textContent = '로그인';
        }
    }

    async function handleRegister(event) {
        event.preventDefault();
        const username = el('registerUsername').value.trim();
        const name = el('registerName').value.trim();
        const password = el('registerPassword').value;
        const passwordConfirm = el('registerPasswordConfirm').value;
        const submit = el('registerSubmit');

        if (!username || !name || !password) {
            setMessage(el('registerMessage'), '모든 항목을 입력해 주세요.');
            return;
        }
        if (password !== passwordConfirm) {
            setMessage(el('registerMessage'), '비밀번호와 비밀번호 확인이 일치하지 않습니다.');
            return;
        }
        submit.disabled = true;
        submit.textContent = '가입 중...';
        try {
            await window.EmotionSession.register({ username, password, passwordConfirm, name });
            goHome();
        } catch (error) {
            setMessage(el('registerMessage'), error.message || '회원가입에 실패했습니다.');
            submit.disabled = false;
            submit.textContent = '회원가입';
        }
    }

    async function handleGuest() {
        const button = el('guestButton');
        button.disabled = true;
        button.textContent = '체험 계정으로 입장 중...';
        try {
            await window.EmotionSession.guestLogin();
            goHome();
        } catch (error) {
            setMessage(el('loginMessage'), error.message || '체험 모드 입장에 실패했습니다.');
            button.disabled = false;
            button.textContent = '로그인 없이 둘러보기 (체험)';
        }
    }

    let usernameCheckTimer = null;
    function scheduleUsernameCheck() {
        const hint = el('usernameHint');
        const username = el('registerUsername').value.trim();
        hint.classList.remove('auth-hint-ok', 'auth-hint-error');
        hint.textContent = '';
        if (usernameCheckTimer) clearTimeout(usernameCheckTimer);
        if (username.length < 4) return;
        usernameCheckTimer = setTimeout(async () => {
            try {
                const available = await window.EmotionSession.checkUsername(username);
                hint.textContent = available ? '사용 가능한 아이디입니다.' : '이미 사용 중인 아이디입니다.';
                hint.classList.toggle('auth-hint-ok', available);
                hint.classList.toggle('auth-hint-error', !available);
            } catch {
                hint.textContent = '';
            }
        }, 350);
    }

    function checkPasswordMatch() {
        const hint = el('confirmHint');
        const password = el('registerPassword').value;
        const confirm = el('registerPasswordConfirm').value;
        hint.classList.remove('auth-hint-ok', 'auth-hint-error');
        if (!confirm) {
            hint.textContent = '';
            return;
        }
        const matched = password === confirm;
        hint.textContent = matched ? '비밀번호가 일치합니다.' : '비밀번호가 일치하지 않습니다.';
        hint.classList.toggle('auth-hint-ok', matched);
        hint.classList.toggle('auth-hint-error', !matched);
    }

    document.addEventListener('DOMContentLoaded', function () {
        // 이미 로그인되어 있으면 메인으로.
        if (window.EmotionSession?.isAuthenticated?.()) {
            goHome();
            return;
        }
        document.querySelectorAll('.auth-tab').forEach((button) => {
            button.addEventListener('click', () => switchTab(button.dataset.tab));
        });
        el('loginForm').addEventListener('submit', handleLogin);
        el('registerForm').addEventListener('submit', handleRegister);
        el('guestButton').addEventListener('click', handleGuest);
        el('registerUsername').addEventListener('input', scheduleUsernameCheck);
        el('registerPassword').addEventListener('input', checkPasswordMatch);
        el('registerPasswordConfirm').addEventListener('input', checkPasswordMatch);
    });
})();
