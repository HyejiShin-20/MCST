// =========================================================
// 프런트엔드 → 백엔드 주소 설정
// =========================================================
// 배포(Vercel) 시 백엔드(Render) 주소를 여기서 지정한다.
// 이 값이 없으면 로컬 개발용 http://127.0.0.1:8000 으로 폴백된다.
//
// ⚠️ 아래 URL을 Render에서 발급된 실제 주소로 교체하세요.
// 각 HTML <head> 에서 session.js / main.js 보다 "먼저" 로드되어야 함:
//   <script src="config.js"></script>
window.EMOTION_API_BASE = "https://mcst-backend.onrender.com";
