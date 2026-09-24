const chatWindow = document.getElementById('chat-window');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question');
const retrievalInfo = document.getElementById('retrieval-info');
const sourcesBox = document.getElementById('sources');
const logsBox = document.getElementById('logs');

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'You' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  if (role === 'user') {
    div.appendChild(bubble);
    div.appendChild(avatar);
  } else {
    div.appendChild(avatar);
    div.appendChild(bubble);
  }

  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function escapeHtml(value = '') {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function renderDebug(data) {
  const debug = data.retrieval_debug || {};
  const dbProof = data.db_proof || {};

  retrievalInfo.innerHTML = `
    <div class="proof-item">
      <strong>Database</strong>
      <small>${dbProof.database || 'storage/hr_policy_chunks.db'}</small>
      <small>Rows found: ${dbProof.rows_found || 0}</small>
    </div>
    <div class="proof-item">
      <strong>Index</strong>
      <small>${debug.index || 'N/A'}</small>
    </div>
    <div class="proof-item">
      <strong>Namespace</strong>
      <small>${debug.namespace || 'N/A'}</small>
    </div>
    <div class="proof-item">
      <strong>Embedding model</strong>
      <small>${debug.embedding_model || 'N/A'}</small>
    </div>
    <div class="proof-item">
      <strong>Top K / Vector dimension</strong>
      <small>${debug.top_k || 0} / ${debug.vector_dimension || 'N/A'}</small>
    </div>
  `;

  sourcesBox.innerHTML = '';
  const sources = data.sources || [];

  if (sources.length === 0) {
    sourcesBox.innerHTML = '<p class="muted">No retrieved chunks.</p>';
  } else {
    sources.forEach((source, index) => {
      const card = document.createElement('div');
      card.className = 'source-card';
      card.innerHTML = `
        <div class="source-head">
          <strong>#${index + 1}</strong>
          <span>score ${Number(source.score || 0).toFixed(4)}</span>
        </div>
        <div class="source-meta">ID: ${escapeHtml(source.id || 'N/A')}</div>
        <div class="source-meta">Page: ${escapeHtml(source.page_number || 'N/A')}</div>
        <div class="source-meta">Parent: ${escapeHtml(source.parent_title || 'N/A')}</div>
        <div class="source-meta">Section: ${escapeHtml(source.section_title || 'N/A')}</div>
        <details>
          <summary>View chunk</summary>
          <pre>${escapeHtml(source.child_text || '')}</pre>
        </details>
      `;
      sourcesBox.appendChild(card);
    });
  }

  logsBox.textContent = (data.logs || ['No logs yet.']).join('\n');
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  addMessage('user', question);
  questionInput.value = '';
  questionInput.focus();

  const typing = document.createElement('div');
  typing.className = 'message bot';
  typing.innerHTML = '<div class="avatar">AI</div><div class="bubble">Thinking...</div>';
  chatWindow.appendChild(typing);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();
    const answer = data.answer || 'No answer available.';

    if (chatWindow.lastElementChild && chatWindow.lastElementChild.querySelector('.bubble')?.textContent === 'Thinking...') {
      chatWindow.removeChild(chatWindow.lastElementChild);
    }

    addMessage('bot', answer);
    renderDebug(data);
  } catch (error) {
    if (chatWindow.lastElementChild && chatWindow.lastElementChild.querySelector('.bubble')?.textContent === 'Thinking...') {
      chatWindow.removeChild(chatWindow.lastElementChild);
    }
    addMessage('bot', 'The backend is unavailable right now. Please check the Flask server and API route.');
    renderDebug({
      retrieval_debug: {},
      db_proof: {},
      sources: [],
      logs: ['API call failed', String(error)],
    });
  }
});
