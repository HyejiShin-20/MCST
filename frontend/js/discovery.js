function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function ratioText(contentRatio) {
    return Object.entries(contentRatio || {})
        .filter(([, value]) => value.count)
        .map(([key, value]) => `${key.toUpperCase()} ${value.percent}%`)
        .join(' / ') || '저장 데이터 없음';
}

function renderNetwork(tags) {
    const svg = document.querySelector('.network-svg');
    if (!svg) return;
    const safeTags = (tags || []).slice(0, 4);
    const points = [
        { x: 150, y: 150 },
        { x: 350, y: 150 },
        { x: 350, y: 350 },
        { x: 250, y: 380 }
    ];
    const lines = points.map((point, index) => `
        <line x1="250" y1="250" x2="${point.x}" y2="${point.y}" stroke="#000" stroke-width="1" ${index % 2 ? '' : 'stroke-dasharray="5,5"'}/>
    `).join('');
    const nodes = points.map((point, index) => `
        <circle cx="${point.x}" cy="${point.y}" r="${safeTags[index] ? 9 : 5}" fill="#000"/>
        <text x="${point.x + 10}" y="${point.y - 10}" font-size="12" font-weight="bold" fill="#000">${escapeHtml(safeTags[index] || '-')}</text>
    `).join('');
    svg.innerHTML = `
        <circle cx="250" cy="250" r="80" fill="#ffffcc" stroke="#000" stroke-width="2"/>
        ${lines}
        ${nodes}
        <circle cx="250" cy="250" r="6" fill="#000"/>
        <text x="205" y="255" font-size="13" font-weight="bold" fill="#000">SAVED TASTE</text>
    `;
}

function renderPanel(data) {
    const summary = data.taste_summary || {};
    const patterns = data.emotion_patterns || {};
    const direction = data.recommendation_direction || {};
    const report = data.generated_report || {};
    const savedCount = data.saved_count || 0;
    const contentRatio = data.content_ratio || {};
    const title = document.querySelector('.discovery-title');
    const statusTitle = document.querySelector('.status-title');
    const overlapValue = document.querySelector('.overlap-value');
    const overlapLabel = document.querySelector('.overlap-label');
    const overlapTags = document.querySelector('.overlap-tags');
    const sectionTitles = document.querySelectorAll('.discovery-panel .section-title');
    const densityInfo = document.querySelector('.density-info');
    const nodeStats = document.querySelectorAll('.node-stat');

    if (title) title.textContent = savedCount ? '저장 취향 분석' : '취향 데이터 대기';
    if (statusTitle) statusTitle.textContent = `SAVED TASTE // ${savedCount} ITEMS`;
    if (overlapValue) overlapValue.textContent = savedCount ? `${Math.min(99, 40 + savedCount * 8)}%` : '0%';
    if (overlapLabel) overlapLabel.textContent = '취향 신호 밀도';
    if (overlapTags) overlapTags.textContent = (summary.top_tags || []).slice(0, 3).join(' / ') || '저장 항목 필요';
    if (sectionTitles[0]) sectionTitles[0].textContent = '콘텐츠 분야 비율';
    if (densityInfo) densityInfo.textContent = ratioText(contentRatio);
    if (nodeStats[0]) {
        nodeStats[0].querySelector('.node-label').textContent = '저장';
        nodeStats[0].querySelector('.node-value').textContent = String(savedCount);
    }
    if (nodeStats[1]) {
        nodeStats[1].querySelector('.node-label').textContent = '태그';
        nodeStats[1].querySelector('.node-value').textContent = String((summary.top_tags || []).length);
    }

    const panel = document.querySelector('.discovery-panel');
    const existing = document.getElementById('tasteAnalysisDetails');
    existing?.remove();
    const details = document.createElement('div');
    details.id = 'tasteAnalysisDetails';
    details.className = 'panel-section system-section';
    details.innerHTML = `
        <div class="section-title">취향 요약</div>
        <p class="system-text">${escapeHtml(summary.headline || '저장된 항목이 생기면 취향 요약을 계산합니다.')}</p>
        <div class="section-title">감정 패턴</div>
        <p class="system-text">${escapeHtml(patterns.summary || '')}</p>
        <div class="section-title">추천 방향</div>
        <p class="system-text">${escapeHtml(direction.next_direction || '')}</p>
        <div class="section-title">생성 리포트</div>
        <p class="system-text">${escapeHtml(report.text || '')}</p>
    `;
    panel?.appendChild(details);
    renderNetwork(summary.top_tags || []);
}

async function loadTasteAnalysis() {
    const userId = window.EmotionSession?.currentUserId?.() || 1;
    const response = await window.EmotionSession.requestJson(`/api/taste-analysis?user_id=${userId}&use_openai=true`);
    renderPanel(response);
}

document.addEventListener('DOMContentLoaded', function() {
    if (!window.EmotionSession?.requireAuth?.()) return;
    window.EmotionSession?.bindUserPanel?.();
    window.EmotionSession?.ensureUser?.().finally(loadTasteAnalysis);
});
