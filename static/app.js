const input = document.getElementById('query');
const tabButtons = document.querySelectorAll('.chip-tab');
const chipGroups = document.querySelectorAll('.chip-group');
const form = document.getElementById('ask-form');
const copyBtn = document.querySelector('.copy-answer');
const answerBlock = document.getElementById('answer-block');
const answerPanel = document.getElementById('answer-panel');
const moreBtn = document.getElementById('more-suggestions');
const dynamicSuggestions = document.getElementById('dynamic-suggestions');

document.addEventListener('click', (event) => {
  const chip = event.target.closest('.chip');
  if (!chip || !input) return;
  input.value = chip.dataset.q || '';
  input.focus();
});

tabButtons.forEach((tab) => {
  tab.addEventListener('click', () => {
    const key = tab.dataset.tab;
    if (!key) return;

    tabButtons.forEach((btn) => {
      const active = btn === tab;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    });

    chipGroups.forEach((group) => {
      group.classList.toggle('active', group.dataset.group === key);
    });
  });
});

if (moreBtn && dynamicSuggestions) {
  moreBtn.addEventListener('click', async () => {
    moreBtn.disabled = true;
    const old = moreBtn.textContent;
    moreBtn.textContent = 'جاري التحميل...';

    try {
      const resp = await fetch('/api/suggest-more', { method: 'GET' });
      if (!resp.ok) throw new Error('bad-status');
      const data = await resp.json();
      const items = Array.isArray(data.items) ? data.items : [];

      if (!items.length) {
        dynamicSuggestions.innerHTML = '<span class="muted-inline">لا توجد أسئلة إضافية حالياً.</span>';
      } else {
        dynamicSuggestions.innerHTML = items
          .slice(0, 12)
          .map((q) => `<button type="button" class="chip" data-q="${q}">${q}</button>`)
          .join('');
      }
    } catch (_) {
      dynamicSuggestions.innerHTML = '<span class="muted-inline">تعذر تحميل الأسئلة الآن. حاول مرة أخرى.</span>';
    } finally {
      moreBtn.disabled = false;
      moreBtn.textContent = old;
    }
  });
}

if (form) {
  form.addEventListener('submit', () => {
    const btn = form.querySelector('button[type="submit"]');
    if (btn) {
      btn.textContent = 'جاري التحليل...';
      btn.disabled = true;
    }
  });
}

if (copyBtn && answerBlock) {
  copyBtn.addEventListener('click', async () => {
    const text = (answerBlock.innerText || '').trim();
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      const old = copyBtn.textContent;
      copyBtn.textContent = 'تم النسخ';
      setTimeout(() => {
        copyBtn.textContent = old;
      }, 1200);
    } catch (_) {
      copyBtn.textContent = 'انسخ يدويًا';
    }
  });
}

if (answerPanel && answerBlock && (answerBlock.innerText || '').trim()) {
  window.requestAnimationFrame(() => {
    answerPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    answerPanel.classList.add('reveal');
  });
}
