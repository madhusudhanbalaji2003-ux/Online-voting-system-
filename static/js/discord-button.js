/**
 * Discord Button Letter-Animation System
 * From Uiverse.io by KINGFRESS
 * White & Black Theme
 * Automatically wraps button text into .span-mother and .span-mother2 layers
 */
(function () {
  'use strict';

  function setupDiscordButton(btn) {
    if (!btn || btn.dataset.discordInit === 'true') return;
    if (btn.querySelector('.span-mother')) {
      btn.dataset.discordInit = 'true';
      return;
    }

    // Exclude internal widgets / calendar / icon-only non-text buttons
    if (
      btn.classList.contains('taskbar-hamburger') ||
      btn.classList.contains('scroll-advance-btn') ||
      btn.classList.contains('mdtp-mode-tab') ||
      btn.classList.contains('mdtp-nav-btn') ||
      btn.classList.contains('mdtp-ampm') ||
      btn.classList.contains('mdtp-cell') ||
      btn.classList.contains('btn-icon') ||
      btn.classList.contains('clock-btn') ||
      btn.querySelector('.spinner')
    ) {
      return;
    }

    // Extract any existing SVG icons
    const svgs = Array.from(btn.querySelectorAll('svg')).map(s => s.cloneNode(true));

    // Extract text content cleanly
    let text = '';
    btn.childNodes.forEach(node => {
      if (node.nodeType === Node.TEXT_NODE) {
        text += node.textContent;
      } else if (node.nodeType === Node.ELEMENT_NODE && node.tagName.toLowerCase() !== 'svg') {
        text += node.textContent;
      }
    });

    text = text.trim();
    if (!text) return;

    btn.dataset.discordInit = 'true';
    btn.innerHTML = '';

    // Wrapper container to preserve layout and allow sibling icons to remain stable
    const wrap = document.createElement('span');
    wrap.className = 'discord-text-wrap';

    // First layer: drops down on hover
    const mother1 = document.createElement('span');
    mother1.className = 'span-mother';

    // Second layer: slides down from above into view on hover
    const mother2 = document.createElement('span');
    mother2.className = 'span-mother2';

    Array.from(text).forEach((char, i) => {
      const s1 = document.createElement('span');
      const s2 = document.createElement('span');

      if (char === ' ') {
        s1.innerHTML = '&nbsp;';
        s2.innerHTML = '&nbsp;';
        s1.style.width = '0.3em';
        s2.style.width = '0.3em';
      } else {
        s1.textContent = char;
        s2.textContent = char;
      }

      // Progressive transition timing per letter matching KINGFRESS effect
      const duration = (0.2 + i * 0.05).toFixed(2) + 's';
      s1.style.transition = 'transform ' + duration + ' cubic-bezier(0.2, 0, 0, 1)';
      s2.style.transition = 'transform ' + duration + ' cubic-bezier(0.2, 0, 0, 1)';

      mother1.appendChild(s1);
      mother2.appendChild(s2);
    });

    wrap.appendChild(mother1);
    wrap.appendChild(mother2);
    btn.appendChild(wrap);

    // Re-attach any SVG icons
    svgs.forEach(svg => {
      btn.appendChild(svg);
    });
  }

  function initAllDiscordButtons() {
    const selector = [
      'button:not(.taskbar-hamburger):not(.mdtp-mode-tab):not(.mdtp-nav-btn):not(.mdtp-ampm):not(.mdtp-cell):not(.scroll-advance-btn)',
      '.btn',
      '.nav-cta',
      '.voter-submit-btn',
      '.voter-cam-btn'
    ].join(',');

    document.querySelectorAll(selector).forEach(setupDiscordButton);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAllDiscordButtons);
  } else {
    initAllDiscordButtons();
  }

  // Handle dynamically inserted or altered buttons
  let debounceTimer = null;
  const observer = new MutationObserver(function () {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(initAllDiscordButtons, 50);
  });

  if (document.body) {
    observer.observe(document.body, { childList: true, subtree: true });
  } else {
    document.addEventListener('DOMContentLoaded', function () {
      observer.observe(document.body, { childList: true, subtree: true });
    });
  }
})();
