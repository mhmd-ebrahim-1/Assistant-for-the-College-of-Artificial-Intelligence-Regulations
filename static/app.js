const input = document.getElementById('query');
const form = document.getElementById('ask-form');
const copyBtn = document.querySelector('.copy-answer');
const answerBlock = document.getElementById('answer-block');
const answerPanel = document.getElementById('answer-panel');
const tabButtons = Array.from(document.querySelectorAll('.chip-tab'));
const tabGroups = Array.from(document.querySelectorAll('.chip-group'));

document.addEventListener('click', (event) => {
  const chip = event.target.closest('.chip');
  if (!chip || !input) return;
  input.value = chip.dataset.q || '';
  input.focus();
});

tabButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    tabButtons.forEach((b) => b.classList.toggle('active', b === btn));
    tabGroups.forEach((group) => {
      group.classList.toggle('active', group.dataset.group === target);
    });
  });
});

if (form) {
  form.addEventListener('submit', () => {
    const btn = form.querySelector('button[type="submit"]');
    if (btn) {
      btn.textContent = 'جاري التحليل...';
      btn.disabled = true;
    }
  });
}

if (input && form) {
  input.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      form.requestSubmit();
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
