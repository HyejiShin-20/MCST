const API_BASE = window.EmotionSession?.API_BASE || window.EMOTION_API_BASE || 'http://127.0.0.1:8000';
const DEFAULT_USER_ID = 1;
const CONTENT_TYPES = ['music', 'book', 'movie'];

let currentMediaId = null;
let currentMediaCandidate = null;
let currentMediaItems = [];
let mediaRecorder = null;
let audioChunks = [];
let drawingInitialized = false;
let drawingHasInk = false;

function activeUserId() {
    return window.EmotionSession?.currentUserId?.() || DEFAULT_USER_ID;
}

function setStatus(message, isError = false) {
    const status = document.getElementById('apiStatus');
    if (!status) return;
    status.textContent = message || '';
    status.style.color = isError ? '#b00020' : '#666';
}

function typeLabel(type) {
    return {
        music: 'MUSIC',
        book: 'BOOK',
        movie: 'MOVIE'
    }[type] || String(type || '').toUpperCase();
}

function flattenRecommendations(response) {
    const grouped = response?.recommendations || {};
    const items = [];
    CONTENT_TYPES.forEach((type) => {
        (grouped[type] || []).forEach((item) => items.push({
            ...item,
            entry_id: item.entry_id || response.entry_id || null,
            log_id: item.log_id || response.log_id || null
        }));
    });
    return items;
}

function updateLastRun(partial) {
    const previous = window.EmotionSession?.loadLastRun?.() || {};
    window.EmotionSession?.saveLastRun?.({
        ...previous,
        ...partial
    });
}

function rememberRecommendation(rec) {
    try {
        localStorage.setItem('emotionCultureRecommendationDetail', JSON.stringify({
            user_id: activeUserId(),
            recommendation: rec,
            saved_at: new Date().toISOString()
        }));
    } catch {
        // Detail fallback can still load content by id.
    }
}

function displayRecommendations(response) {
    const container = document.getElementById('recommendationsList');
    const items = flattenRecommendations(response);
    container.innerHTML = '';

    if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'recommendation-card';
        empty.innerHTML = `
            <div class="rec-header">
                <div class="rec-number">NO RESULT</div>
                <div class="rec-title">추천 결과 없음</div>
            </div>
            <div class="rec-reason">분석은 완료됐지만 조건에 맞는 음악/도서 추천을 찾지 못했습니다.</div>
        `;
        container.appendChild(empty);
        return;
    }

    items.forEach((rec, idx) => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';
        const label = typeLabel(rec.content_type);
        const typeClass = rec.content_type || 'book';
        const tags = (rec.display_tags && rec.display_tags.length)
            ? rec.display_tags.slice(0, 4)
            : [...(rec.emotion_tags || []), ...(rec.topic_tags || [])].slice(0, 4);
        const score = Math.round(rec.score_percent || (Number(rec.score || 0) * 100));

        card.innerHTML = `
            <div class="rec-header">
                <div class="rec-number">RECOMMENDATION ${String(idx + 1).padStart(2, '0')}</div>
                <div class="rec-title">${escapeHtml(rec.title || '제목 없음')}</div>
            </div>
            <div class="rec-tags">
                ${tags.map((tag) => `<span class="rec-tag">#${escapeHtml(tag)}</span>`).join('')}
            </div>
            <div class="rec-reason">${escapeHtml(rec.reason || rec.summary || '추천 이유를 생성하지 못했습니다.')}</div>
            <button type="button" class="rec-save-button">SAVE TO REPOSITORY</button>
            <div class="rec-footer">
                <div class="rec-dots"></div>
                <div class="rec-type ${typeClass}">
                    ${label}
                    <span class="rec-type-label">${escapeHtml(rec.creator || '')}</span>
                </div>
                <div class="rec-score">
                    <span class="rec-score-label">MATCH</span>
                    <span class="rec-score-value">${score}%</span>
                </div>
            </div>
        `;

        const saveButton = card.querySelector('.rec-save-button');
        saveButton?.addEventListener('click', (event) => {
            event.stopPropagation();
            saveRecommendation(rec, saveButton);
        });

        card.addEventListener('click', function() {
            const id = encodeURIComponent(rec.content_id || idx + 1);
            rememberRecommendation(rec);
            window.location.href = `recommendation-detail.html?id=${id}`;
        });

        container.appendChild(card);
    });
}

async function saveRecommendation(rec, button) {
    const tags = (rec.display_tags && rec.display_tags.length)
        ? rec.display_tags
        : [...(rec.emotion_tags || []), ...(rec.topic_tags || [])].slice(0, 8);
    const score = Math.round(rec.score_percent || (Number(rec.score || 0) * 100));
    button.disabled = true;
    button.textContent = 'SAVING...';
    try {
        await requestJson('/api/repository/save', {
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
                score_percent: score,
                display_tags: tags,
                status: 'planned',
                snapshot: rec
            })
        });
        button.textContent = 'SAVED';
        setStatus('저장소에 추천 항목을 저장했습니다.');
    } catch (error) {
        button.disabled = false;
        button.textContent = 'SAVE TO REPOSITORY';
        setStatus(`저장 실패: ${error.message}`, true);
    }
}

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
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

async function analyzeEmotion() {
    const diaryTextarea = document.querySelector('.daily-entry-textarea');
    const reviewBox = document.getElementById('transcriptionReview');
    const transcriptionText = document.getElementById('transcriptionText');
    const reviewedText = transcriptionText?.value?.trim() || '';
    const diaryText = diaryTextarea.value.trim();
    const hasReviewedMedia = currentMediaItems.length > 0 && reviewedText && reviewBox && !reviewBox.hidden;
    const text = [diaryText, hasReviewedMedia ? reviewedText : ''].filter(Boolean).join('\n\n');

    if (!text) {
        alert('먼저 일기를 쓰거나 파일을 업로드해 주세요.');
        return;
    }

    setStatus('분석과 추천을 요청하는 중입니다.');
    try {
        let result;
        if (hasReviewedMedia) {
            result = await requestJson('/api/diary/from-transcription', {
                method: 'POST',
                body: JSON.stringify({
                    user_id: activeUserId(),
                    media_id: currentMediaItems[0]?.media_id || currentMediaId,
                    media_items: currentMediaItems.map((item) => ({
                        media_id: item.media_id,
                        selected_text: item.selected_text || ''
                    })),
                    selected_text: reviewedText,
                    context_text: diaryText,
                    content_types: CONTENT_TYPES,
                    per_type: 4,
                    return_recommendations: true
                })
            });
            displayRecommendations(result.recommendations);
            window.EmotionSession?.saveLastRun?.({
                entry_id: result.diary?.entry_id,
                source_text: result.raw_text || text,
                diary_draft: diaryText,
                media_text: reviewedText,
                analysis: result.diary?.analysis,
                media_transcription: result.media_transcription,
                media_transcriptions: result.media_transcriptions || currentMediaItems.map((item) => item.media),
                recommendations: result.recommendations
            });
        } else {
            const diary = await requestJson('/api/diary', {
                method: 'POST',
                body: JSON.stringify({
                    user_id: activeUserId(),
                    input_type: 'text',
                    text
                })
            });
            result = await requestJson('/api/recommend', {
                method: 'POST',
                body: JSON.stringify({
                    entry_id: diary.entry_id,
                    content_types: CONTENT_TYPES,
                    per_type: 4
                })
            });
            displayRecommendations(result);
            window.EmotionSession?.saveLastRun?.({
                entry_id: diary.entry_id,
                source_text: text,
                diary_draft: diaryText,
                media_text: '',
                media_transcriptions: [],
                analysis: diary.analysis,
                recommendations: result
            });
        }
        setStatus('추천 결과를 업데이트했습니다.');
        document.querySelector('.recommendation-column').scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        setStatus(`요청 실패: ${error.message}`, true);
    }
}

function setMediaCandidate(media, sourceLabel) {
    addMediaCandidate(media, sourceLabel);
}

function mediaCandidateText(media) {
    return media?.selected_text
        || (media?.candidates && media.candidates[0] && media.candidates[0].text)
        || '';
}

function mediaLabel(item) {
    if (item.source_label) return item.source_label;
    return {
        image: '이미지',
        audio: '음성',
        link: '링크'
    }[item.input_type] || '첨부';
}

function mediaItemFromCandidate(media, sourceLabel) {
    const mediaId = Number(media?.media_id || 0);
    return {
        media_id: mediaId,
        input_type: media?.input_type || 'media',
        source_label: sourceLabel || '',
        selected_text: mediaCandidateText(media),
        media
    };
}

function formatMediaBlock(item) {
    const label = mediaLabel(item);
    const text = item.selected_text || mediaCandidateText(item.media);
    return `[${label} #${item.media_id}]\n${text}`.trim();
}

function ensureTranscriptionItemsContainer() {
    let container = document.getElementById('transcriptionItems');
    if (container) return container;
    const reviewBox = document.getElementById('transcriptionReview');
    const textArea = document.getElementById('transcriptionText');
    if (!reviewBox || !textArea) return null;
    container = document.createElement('div');
    container.id = 'transcriptionItems';
    container.className = 'transcription-items';
    reviewBox.insertBefore(container, textArea);
    return container;
}

function renderMediaItems() {
    const container = ensureTranscriptionItemsContainer();
    if (!container) return;
    container.innerHTML = '';
    currentMediaItems.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'transcription-item';
        row.innerHTML = `
            <span class="transcription-item-meta">${escapeHtml(mediaLabel(item))} #${escapeHtml(item.media_id)}</span>
            <button type="button" class="transcription-item-remove" aria-label="remove media">REMOVE</button>
        `;
        row.querySelector('.transcription-item-remove')?.addEventListener('click', () => {
            removeMediaCandidate(item.media_id);
        });
        container.appendChild(row);
    });
}

function persistMediaDraft() {
    const diaryText = document.querySelector('.daily-entry-textarea')?.value?.trim() || '';
    const mediaText = document.getElementById('transcriptionText')?.value || '';
    const lastItem = currentMediaItems[currentMediaItems.length - 1] || null;
    updateLastRun({
        media_transcription: lastItem?.media || null,
        media_transcriptions: currentMediaItems.map((item) => item.media),
        media_text: mediaText,
        diary_draft: diaryText,
        recommendations: null
    });
}

function rebuildTranscriptionTextFromItems() {
    const transcriptionText = document.getElementById('transcriptionText');
    if (!transcriptionText) return;
    transcriptionText.value = currentMediaItems.map(formatMediaBlock).filter(Boolean).join('\n\n');
}

function addMediaCandidate(media, sourceLabel) {
    const item = mediaItemFromCandidate(media, sourceLabel);
    if (!item.media_id) return;
    const existingIndex = currentMediaItems.findIndex((candidate) => candidate.media_id === item.media_id);
    if (existingIndex >= 0) {
        currentMediaItems[existingIndex] = item;
        rebuildTranscriptionTextFromItems();
    } else {
        currentMediaItems.push(item);
        const transcriptionText = document.getElementById('transcriptionText');
        const block = formatMediaBlock(item);
        if (transcriptionText && block) {
            const currentText = transcriptionText.value.trim();
            transcriptionText.value = currentText ? `${currentText}\n\n${block}` : block;
        }
    }
    currentMediaId = item.media_id;
    currentMediaCandidate = item.media;
    const reviewBox = document.getElementById('transcriptionReview');
    if (reviewBox) reviewBox.hidden = false;
    renderMediaItems();
    setStatus(`${sourceLabel} 변환 텍스트를 추가했습니다. 필요한 경우 통합 텍스트를 수정한 뒤 감정 분석을 눌러 주세요.`);
    persistMediaDraft();
}

function removeMediaCandidate(mediaId) {
    currentMediaItems = currentMediaItems.filter((item) => item.media_id !== Number(mediaId));
    const lastItem = currentMediaItems[currentMediaItems.length - 1] || null;
    currentMediaId = lastItem?.media_id || null;
    currentMediaCandidate = lastItem?.media || null;
    rebuildTranscriptionTextFromItems();
    renderMediaItems();
    const reviewBox = document.getElementById('transcriptionReview');
    if (reviewBox && currentMediaItems.length === 0) reviewBox.hidden = true;
    persistMediaDraft();
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const result = String(reader.result || '');
            resolve(result.includes(',') ? result.split(',').pop() : result);
        };
        reader.onerror = () => reject(reader.error || new Error('파일을 읽지 못했습니다.'));
        reader.readAsDataURL(file);
    });
}

async function uploadMediaBlob(inputType, blob, filename, mimeType, sourceLabel) {
    const formData = new FormData();
    formData.append('user_id', String(activeUserId()));
    formData.append('input_type', inputType);
    formData.append('file', blob, filename);

    setStatus(`${sourceLabel} 텍스트 변환 중입니다.`);
    try {
        let response;
        try {
            response = await requestJson('/api/media/transcribe', {
                method: 'POST',
                body: formData
            });
        } catch (multipartError) {
            const file = blob instanceof File ? blob : new File([blob], filename, { type: mimeType });
            const dataBase64 = await fileToBase64(file);
            response = await requestJson('/api/media/transcribe-base64', {
                method: 'POST',
                body: JSON.stringify({
                    user_id: activeUserId(),
                    input_type: inputType,
                    filename,
                    mime_type: mimeType || file.type || '',
                    data_base64: dataBase64
                })
            });
        }
        setMediaCandidate(response.media_transcription, sourceLabel);
    } catch (error) {
        setStatus(`${sourceLabel} 변환 실패: ${error.message}`, true);
    }
}

async function uploadMedia(inputType, fileInput) {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    const sourceLabel = inputType === 'audio' ? '음성' : '이미지';
    try {
        await uploadMediaBlob(inputType, file, file.name || `${inputType}-upload`, file.type || '', sourceLabel);
    } finally {
        fileInput.value = '';
    }
}

async function handleAudioOptionClick(event) {
    event.preventDefault();
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        return;
    }
    if (!navigator.mediaDevices || !window.MediaRecorder) {
        document.getElementById('audioInput')?.click();
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];
        const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
        mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size) audioChunks.push(event.data);
        };
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach((track) => track.stop());
            document.getElementById('audioAction').textContent = '녹음 시작';
            const type = mediaRecorder.mimeType || 'audio/webm';
            const blob = new Blob(audioChunks, { type });
            if (!blob.size) {
                setStatus('녹음된 음성이 없습니다.', true);
                return;
            }
            await uploadMediaBlob('audio', blob, `recording-${Date.now()}.webm`, type, '음성');
        };
        mediaRecorder.start();
        document.getElementById('audioAction').textContent = '녹음 중지';
        setStatus('녹음 중입니다. 음성 칸을 다시 누르면 변환합니다.');
    } catch (error) {
        setStatus(`녹음 시작 실패: ${error.message}`, true);
        document.getElementById('audioInput')?.click();
    }
}

function openDrawingPanel() {
    const panel = document.getElementById('drawingPanel');
    panel.hidden = !panel.hidden;
    if (!panel.hidden) setupDrawingCanvas();
}

function setupDrawingCanvas() {
    if (drawingInitialized) return;
    drawingInitialized = true;
    const canvas = document.getElementById('drawingCanvas');
    const context = canvas.getContext('2d');
    context.fillStyle = '#fff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = '#000';
    context.lineWidth = 4;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    let drawing = false;

    const point = (event) => {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (event.clientX - rect.left) * (canvas.width / rect.width),
            y: (event.clientY - rect.top) * (canvas.height / rect.height)
        };
    };
    canvas.addEventListener('pointerdown', (event) => {
        drawing = true;
        drawingHasInk = true;
        canvas.setPointerCapture(event.pointerId);
        const p = point(event);
        context.beginPath();
        context.moveTo(p.x, p.y);
    });
    canvas.addEventListener('pointermove', (event) => {
        if (!drawing) return;
        const p = point(event);
        context.lineTo(p.x, p.y);
        context.stroke();
    });
    canvas.addEventListener('pointerup', () => {
        drawing = false;
    });
    canvas.addEventListener('pointercancel', () => {
        drawing = false;
    });
}

function clearDrawing() {
    setupDrawingCanvas();
    const canvas = document.getElementById('drawingCanvas');
    const context = canvas.getContext('2d');
    context.fillStyle = '#fff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    drawingHasInk = false;
}

async function submitDrawing() {
    setupDrawingCanvas();
    if (!drawingHasInk) {
        setStatus('그림을 먼저 그려 주세요.', true);
        return;
    }
    const canvas = document.getElementById('drawingCanvas');
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    if (!blob) {
        setStatus('그림을 이미지로 변환하지 못했습니다.', true);
        return;
    }
    await uploadMediaBlob('image', blob, `drawing-${Date.now()}.png`, 'image/png', '그림');
}

function openLinkPanel() {
    const panel = document.getElementById('linkPanel');
    panel.hidden = !panel.hidden;
    if (!panel.hidden) document.getElementById('linkInput')?.focus();
}

async function submitLink() {
    const input = document.getElementById('linkInput');
    const url = input.value.trim();
    if (!url) {
        setStatus('링크를 입력해 주세요.', true);
        return;
    }
    setStatus('링크 텍스트를 가져오는 중입니다.');
    try {
        const response = await requestJson('/api/link/transcribe', {
            method: 'POST',
            body: JSON.stringify({
                user_id: activeUserId(),
                url
            })
        });
        setMediaCandidate(response.media_transcription, '링크');
    } catch (error) {
        setStatus(`링크 변환 실패: ${error.message}`, true);
    }
}

function restoreLastRun() {
    const lastRun = window.EmotionSession?.loadLastRun?.();
    if (!lastRun) return;
    const diaryTextarea = document.querySelector('.daily-entry-textarea');
    if (diaryTextarea && !diaryTextarea.value) {
        const hasMedia = (lastRun.media_transcriptions && lastRun.media_transcriptions.length)
            || lastRun.media_transcription;
        const draftText = lastRun.diary_draft || (!hasMedia ? lastRun.source_text : '');
        if (draftText) diaryTextarea.value = draftText;
    }
    const storedMedia = Array.isArray(lastRun.media_transcriptions) && lastRun.media_transcriptions.length
        ? lastRun.media_transcriptions
        : (lastRun.media_transcription ? [lastRun.media_transcription] : []);
    if (storedMedia.length) {
        currentMediaItems = storedMedia
            .filter((media) => media && media.media_id)
            .map((media) => mediaItemFromCandidate(media, ''));
        const lastItem = currentMediaItems[currentMediaItems.length - 1] || null;
        currentMediaId = lastItem?.media_id || null;
        currentMediaCandidate = lastItem?.media || null;
        const transcriptionText = document.getElementById('transcriptionText');
        const reviewBox = document.getElementById('transcriptionReview');
        if (transcriptionText) {
            transcriptionText.value = lastRun.media_text || currentMediaItems.map(formatMediaBlock).join('\n\n');
        }
        renderMediaItems();
        if (reviewBox) reviewBox.hidden = false;
    }
    if (lastRun.recommendations) {
        displayRecommendations(lastRun.recommendations);
        setStatus('이전 분석 결과를 복원했습니다.');
    }
}

document.addEventListener('DOMContentLoaded', function() {
    if (!window.EmotionSession?.requireAuth?.()) return;

    const audioInput = document.getElementById('audioInput');
    const photoInput = document.getElementById('photoInput');
    const linkInput = document.getElementById('linkInput');
    const transcriptionText = document.getElementById('transcriptionText');
    const diaryTextarea = document.querySelector('.daily-entry-textarea');

    window.EmotionSession?.bindUserPanel?.();
    window.EmotionSession?.ensureUser?.();
    audioInput?.addEventListener('change', () => uploadMedia('audio', audioInput));
    photoInput?.addEventListener('change', () => uploadMedia('image', photoInput));
    diaryTextarea?.addEventListener('input', () => {
        updateLastRun({
            diary_draft: diaryTextarea.value,
            source_text: diaryTextarea.value
        });
    });
    transcriptionText?.addEventListener('input', () => {
        if (currentMediaItems.length) persistMediaDraft();
    });
    restoreLastRun();
    linkInput?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') submitLink();
    });
    setStatus('일기를 쓰거나 파일을 업로드하면 음악과 도서를 추천합니다.');
});
