let currentTypeFilter = 'all';
let currentStatusFilter = 'all';

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function typeLabel(type) {
    return { music: '음악', book: '도서', movie: '영화' }[type] || String(type || '').toUpperCase();
}

function statusLabel(status) {
    return { planned: '저장됨', completed: '완료', archived: '보관' }[status] || status || '저장됨';
}

function setButtonGroup(buttons, labels, onClick) {
    buttons.forEach((button, index) => {
        const config = labels[index];
        if (!config) return;
        button.textContent = config.label;
        button.dataset.value = config.value;
        button.addEventListener('click', () => {
            buttons.forEach((item) => item.classList.remove('filter-btn-active'));
            button.classList.add('filter-btn-active');
            onClick(config.value);
        });
    });
}

function renderItems(items) {
    const grid = document.getElementById('repositoryGrid') || document.querySelector('.items-grid');
    if (!grid) return;
    if (!items.length) {
        grid.innerHTML = `
            <div class="item-card">
                <div class="item-header">
                    <span class="item-type music">EMPTY</span>
                    <span class="item-status unwatched">READY</span>
                </div>
                <h3 class="item-title">저장된 추천 항목이 없습니다</h3>
                <p class="item-meta">추천 카드의 SAVE 버튼을 누르면 이곳에 보관됩니다.</p>
                <div class="item-tags"><span class="tag">repository</span></div>
            </div>
        `;
        return;
    }
    grid.innerHTML = items.map((item) => {
        const tags = (item.display_tags || []).slice(0, 5);
        const typeClass = item.content_type === 'book' ? 'song' : item.content_type;
        return `
            <div class="item-card" data-saved-id="${item.saved_id}">
                <div class="item-header">
                    <span class="item-type ${escapeHtml(typeClass)}">${escapeHtml(typeLabel(item.content_type))}</span>
                    <span class="item-status ${item.status === 'completed' ? 'watched' : 'unwatched'}">${escapeHtml(statusLabel(item.status))}</span>
                </div>
                <h3 class="item-title">${escapeHtml(item.title)}</h3>
                <p class="item-meta">${escapeHtml(item.creator || item.genre || item.source || '')}</p>
                <p class="item-meta">${escapeHtml(item.reason || '').slice(0, 150)}</p>
                <div class="item-tags">
                    ${tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}
                </div>
                <button type="button" class="item-delete-button" data-delete-id="${item.saved_id}">DELETE</button>
            </div>
        `;
    }).join('');
    grid.querySelectorAll('.item-card').forEach((card, index) => {
        card.addEventListener('click', () => {
            const item = items[index];
            const snapshot = item.snapshot && Object.keys(item.snapshot).length ? item.snapshot : item;
            localStorage.setItem('emotionCultureRecommendationDetail', JSON.stringify({
                user_id: window.EmotionSession?.currentUserId?.() || 1,
                recommendation: {
                    ...snapshot,
                    content_id: item.content_id,
                    content_type: item.content_type,
                    title: item.title,
                    creator: item.creator,
                    genre: item.genre,
                    source: item.source,
                    reason: item.reason || snapshot.reason || '',
                    score_percent: item.score_percent || snapshot.score_percent || 0,
                    display_tags: item.display_tags || snapshot.display_tags || []
                },
                saved_at: new Date().toISOString()
            }));
            window.location.href = `recommendation-detail.html?id=${encodeURIComponent(item.content_id)}`;
        });
    });
    grid.querySelectorAll('.item-delete-button').forEach((button) => {
        button.addEventListener('click', async (event) => {
            event.stopPropagation();
            const savedId = Number(button.dataset.deleteId || 0);
            if (!savedId) return;
            button.disabled = true;
            button.textContent = 'DELETING';
            try {
                await deleteRepositoryItem(savedId);
                await loadRepository();
            } catch (error) {
                button.disabled = false;
                button.textContent = 'DELETE';
                alert(`삭제 실패: ${error.message}`);
            }
        });
    });
}

function updateStats(response) {
    const counts = response.counts || {};
    const values = document.querySelectorAll('.repository-sidebar .stat-value');
    if (values[0]) values[0].textContent = String(counts.total || 0);
    if (values[1]) values[1].textContent = String(counts.planned || 0);
    if (values[2]) values[2].textContent = String(counts.completed || 0);
}

async function loadRepository() {
    const userId = window.EmotionSession?.currentUserId?.() || 1;
    const query = new URLSearchParams({
        user_id: String(userId),
        content_type: currentTypeFilter,
        status: currentStatusFilter,
        limit: '200'
    });
    const response = await window.EmotionSession.requestJson(`/api/repository?${query}`);
    updateStats(response);
    renderItems(response.saved_items || []);
}

async function deleteRepositoryItem(savedId) {
    const userId = window.EmotionSession?.currentUserId?.() || 1;
    await window.EmotionSession.requestJson(`/api/repository/${savedId}?user_id=${userId}`, {
        method: 'DELETE'
    });
}

function setupFilters() {
    const sections = document.querySelectorAll('.repository-sidebar .filter-section');
    const typeButtons = sections[0]?.querySelectorAll('.filter-btn') || [];
    const statusButtons = sections[1]?.querySelectorAll('.filter-btn') || [];
    setButtonGroup(
        Array.from(typeButtons),
        [
            { label: '모두', value: 'all' },
            { label: '음악', value: 'music' },
            { label: '도서', value: 'book' },
            { label: '영화', value: 'movie' }
        ],
        (value) => {
            currentTypeFilter = value;
            loadRepository();
        }
    );
    setButtonGroup(
        Array.from(statusButtons),
        [
            { label: '모두', value: 'all' },
            { label: '저장됨', value: 'planned' },
            { label: '완료', value: 'completed' }
        ],
        (value) => {
            currentStatusFilter = value;
            loadRepository();
        }
    );
}

document.addEventListener('DOMContentLoaded', function() {
    if (!window.EmotionSession?.requireAuth?.()) return;
    setupFilters();
    window.EmotionSession?.bindUserPanel?.();
    window.EmotionSession?.ensureUser?.().finally(loadRepository);
});
