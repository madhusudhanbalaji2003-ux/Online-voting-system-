(function () {
  "use strict";

  // Replaces an empty .sf-module div with a clean, unbroken glowing card
  function buildModule(el) {
    el.innerHTML = '';
    
    var inner = document.createElement('div');
    inner.className = 'sf-card-inner';
    
    var text = document.createElement('span');
    text.className = 'sf-digit-val';
    text.textContent = '-';
    
    inner.appendChild(text);
    el.appendChild(inner);

    return {
      el: el,
      inner: inner,
      text: text,
      current: '-'
    };
  }

  // Registry of all modules by their data-id
  window.sfModules = {};
  window.sfInit = function() {
    var els = document.querySelectorAll('.sf-module');
    els.forEach(function(el) {
      var id = el.getAttribute('data-id');
      if (id && !window.sfModules[id]) {
        window.sfModules[id] = buildModule(el);
      }
    });
  };

  // Set single digit with smooth whole-card 3D flip animation
  window.sfSetDigit = function(id, d) {
    var m = window.sfModules[id];
    if (!m) return;
    if (String(d) === String(m.current)) return;
    
    m.current = String(d);
    m.text.textContent = d;
    
    m.inner.classList.remove('sf-flip-anim');
    void m.inner.offsetWidth; // Force CSS reflow to re-trigger animation
    m.inner.classList.add('sf-flip-anim');
  };

  // Helper to set 2-digit pairs
  window.sfSetUnit = function(prefix, val) {
    var str = String(val).padStart(2, '0');
    window.sfSetDigit(prefix + '1', str[0]);
    window.sfSetDigit(prefix + '2', str[1]);
  };

  // Auto-init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', window.sfInit);
  } else {
    window.sfInit();
  }

})();
