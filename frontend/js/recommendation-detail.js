function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function activeUserId() {
    return window.EmotionSession?.currentUserId?.() || 1;
}

function typeLabel(type) {
    return { music: '음악', book: '도서', movie: '영화' }[type] || String(type || '').toUpperCase();
}

function flattenLastRecommendations(lastRun) {
    const grouped = lastRun?.recommendations?.recommendations || {};
    return Object.values(grouped).flat();
}

function loadStoredRecommendation(contentId) {
    try {
        const detail = JSON.parse(localStorage.getItem('emotionCultureRecommendationDetail') || '{}');
        const rec = detail?.recommendation;
        if (rec && (!contentId || rec.content_id === contentId)) return rec;
    } catch {
        // Fallback to the last recommendation run.
    }
    const lastRun = window.EmotionSession?.loadLastRun?.();
    return flattenLastRecommendations(lastRun).find((item) => item.content_id === contentId) || null;
}

async function loadContentFallback(contentId) {
    if (!contentId || !window.EmotionSession?.requestJson) return null;
    const response = await window.EmotionSession.requestJson(`/api/content/${encodeURIComponent(contentId)}`);
    const content = response.content || {};
    return {
        ...content,
        score_percent: content.score_percent || 0,
        display_tags: [
            ...(content.emotion_tags || []),
            ...(content.topic_tags || []),
            ...(content.mood_tags || [])
        ].slice(0, 5),
        reason: content.summary || '콘텐츠 원본 정보만 확인했습니다. 추천 근거는 직전 추천 기록에서 열면 함께 표시됩니다.'
    };
}

function renderDetail(rec) {
    const score = Math.round(rec.score_percent || (Number(rec.score || 0) * 100));
    const tags = (rec.display_tags && rec.display_tags.length)
        ? rec.display_tags
        : [...(rec.emotion_tags || []), ...(rec.topic_tags || []), ...(rec.mood_tags || [])].slice(0, 5);
    const category = typeLabel(rec.content_type);
    const creator = rec.creator || rec.source || '제작자 정보 없음';
    const summary = rec.summary || rec.reason || '요약 정보가 없습니다.';
    const role = rec.recommendation_role || rec.expected_effect || '';

    document.querySelector('.rec-number').textContent = rec.content_id ? `추천 ${rec.content_id}` : '추천 상세';
    document.querySelector('.category-badge').textContent = `${category} 추천`;
    document.querySelector('.detail-title').textContent = rec.title || '제목 없음';
    document.querySelector('.detail-tags').innerHTML = tags
        .map((tag) => `<span class="detail-tag">${escapeHtml(tag)}</span>`)
        .join('');
    const analysisNodes = document.querySelectorAll('.analysis-section .analysis-text');
    if (analysisNodes[0]) analysisNodes[0].textContent = rec.reason || '추천 이유가 없습니다.';
    if (analysisNodes[1]) analysisNodes[1].textContent = summary;
    const metadataNames = document.querySelectorAll('.metadata-name');
    const metadataSubtitles = document.querySelectorAll('.metadata-subtitle');
    if (metadataNames[0]) metadataNames[0].textContent = creator;
    if (metadataSubtitles[0]) metadataSubtitles[0].textContent = [rec.genre, rec.source].filter(Boolean).join(' / ') || category;
    if (metadataNames[1]) metadataNames[1].textContent = role || '추천 역할';
    if (metadataSubtitles[1]) metadataSubtitles[1].textContent = rec.expected_effect || '현재 기록과의 정서·상황 적합도를 기준으로 산정했습니다.';
    document.querySelector('.match-score').textContent = `${score}%`;
    const fill = document.querySelector('.match-fill');
    if (fill) fill.style.width = `${Math.max(0, Math.min(100, score))}%`;
    const description = document.querySelector('.match-description');
    if (description) {
        description.textContent = '직접 작성 기록, 변환 텍스트, 감정·상황 태그, 콘텐츠 태그와 회복 적합도를 나눠 계산한 점수입니다.';
    }
    const saveButton = document.querySelector('.action-button-primary');
    if (saveButton) {
        saveButton.onclick = () => saveRecommendation(rec, saveButton);
    }
}

async function saveRecommendation(rec, button) {
    button.disabled = true;
    button.textContent = '저장 중';
    try {
        await window.EmotionSession.requestJson('/api/repository/save', {
            method: 'POST',
            body: JSON.stringify({
                user_id: activeUserId(),
                content_id: rec.content_id,
                content_type: rec.content_type,
                title: rec.title || '',
                creator: rec.creator || '',
                genre: rec.genre || '',
                source: rec.source || '',
                entry_id: rec.entry_id || null,
                log_id: rec.log_id || null,
                recommendation_role: rec.recommendation_role || '',
                reason: rec.reason || '',
                score_percent: Math.round(rec.score_percent || (Number(rec.score || 0) * 100)),
                display_tags: rec.display_tags || [],
                status: 'planned',
                snapshot: rec
            })
        });
        button.textContent = '저장됨';
    } catch (error) {
        button.disabled = false;
        button.textContent = `저장 실패`;
        console.error(error);
    }
}

document.addEventListener('DOMContentLoaded', async function() {
    const backLink = document.querySelector('.back-link');
    if (backLink) {
        backLink.addEventListener('click', function(e) {
            e.preventDefault();
            window.history.back();
        });
    }

    document.querySelectorAll('.action-button-secondary').forEach((button) => {
        button.hidden = true;
    });

    const params = new URLSearchParams(window.location.search);
    const contentId = params.get('id') || '';
    try {
        const rec = loadStoredRecommendation(contentId) || await loadContentFallback(contentId);
        if (!rec) throw new Error('추천 정보를 찾지 못했습니다.');
        renderDetail(rec);
    } catch (error) {
        document.querySelector('.detail-title').textContent = '추천 정보를 찾지 못했습니다';
        const analysis = document.querySelector('.analysis-section .analysis-text');
        if (analysis) analysis.textContent = error.message;
    }
});
