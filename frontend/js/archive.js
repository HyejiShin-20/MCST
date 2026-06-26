function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function formatDate(value) {
    if (!value) return '';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('ko-KR');
}

function tagsHtml(tags) {
    return (tags || [])
        .slice(0, 5)
        .map((tag) => `<span class="card-tag">${escapeHtml(tag)}</span>`)
        .join('');
}

function entryEvent(entry) {
    const analysis = entry.analysis || {};
    const structured = analysis.structured_emotion || {};
    const tags = [
        structured.dominant_emotion,
        ...(structured.sub_emotions || []),
        ...(structured.context_keywords || [])
    ].filter(Boolean);
    return {
        kind: 'entry',
        created_at: entry.created_at,
        title: `기록 #${entry.entry_id} · ${entry.input_type}`,
        content: entry.raw_text || entry.processed_text || '',
        tags
    };
}

function mediaEvent(media) {
    const meta = media.metadata || {};
    const tags = [media.input_type, media.provider, media.status].filter(Boolean);
    const fileInfo = [
        media.source_filename,
        media.source_mime_type,
        media.source_size_bytes ? `${media.source_size_bytes} bytes` : ''
    ].filter(Boolean).join(' / ');
    return {
        kind: 'media',
        created_at: media.created_at,
        title: `변환 텍스트 #${media.media_id} · ${media.input_type}`,
        content: `${fileInfo ? `${fileInfo}\n` : ''}${media.selected_text || ''}`,
        tags: meta.temp_file_deleted ? [...tags, 'temp deleted'] : tags
    };
}

function recommendationEvent(log) {
    const titles = (log.top_recommendations || [])
        .slice(0, 4)
        .map((item) => `${item.title}${item.score_percent ? ` ${item.score_percent}%` : ''}`)
        .join(', ');
    return {
        kind: 'recommendation',
        created_at: log.created_at,
        title: `추천 결과 #${log.log_id}`,
        content: titles || '추천 결과가 저장되어 있습니다.',
        tags: (log.response?.recommendation_roles || []).slice(0, 5)
    };
}

function renderTimeline(activity) {
    const container = document.getElementById('archiveTimeline') || document.querySelector('.timeline-container');
    if (!container) return;
    const events = [
        ...(activity.entries || []).map(entryEvent),
        ...(activity.media_transcriptions || []).map(mediaEvent),
        ...(activity.recommendation_logs || []).map(recommendationEvent)
    ].sort((left, right) => new Date(right.created_at) - new Date(left.created_at));

    if (!events.length) {
        container.innerHTML = `
            <div class="timeline-event">
                <div class="timeline-dot"></div>
                <div class="timeline-card">
                    <div class="card-label">EMPTY</div>
                    <h3 class="card-title">아직 저장된 기록이 없습니다</h3>
                    <p class="card-content">일기, 이미지, 음성, 링크를 분석하면 이곳에 변환 텍스트와 추천 로그가 쌓입니다.</p>
                </div>
            </div>
        `;
        return;
    }

    container.innerHTML = events.map((event) => `
        <div class="timeline-event">
            <div class="timeline-dot ${event.kind === 'recommendation' ? 'timeline-dot-yellow' : ''}"></div>
            <div class="timeline-card">
                <div class="card-label">${escapeHtml(formatDate(event.created_at))}</div>
                <h3 class="card-title">${escapeHtml(event.title)}</h3>
                <p class="card-content">${escapeHtml(event.content).replaceAll('\n', '<br>')}</p>
                <div class="card-tags">${tagsHtml(event.tags)}</div>
            </div>
        </div>
    `).join('');
}

function updateSummary(activity) {
    const summary = activity.summary || {};
    const values = document.querySelectorAll('.archive-sidebar .stat-value');
    if (values[0]) values[0].textContent = String(summary.entry_count || 0);
    if (values[1]) values[1].textContent = String(summary.media_count || 0);
    if (values[2]) values[2].textContent = String(summary.recommendation_count || 0);
}

async function loadArchive() {
    const userId = window.EmotionSession?.currentUserId?.() || 1;
    const activity = await window.EmotionSession.requestJson(`/api/activity?user_id=${userId}&limit=100`);
    updateSummary(activity);
    renderTimeline(activity);
}

document.addEventListener('DOMContentLoaded', function() {
    if (!window.EmotionSession?.requireAuth?.()) return;
    window.EmotionSession?.bindUserPanel?.();
    window.EmotionSession?.ensureUser?.().finally(loadArchive);
});
