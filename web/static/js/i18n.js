(function(){
  const I18N_KEY = 'chan_lang';
  let current = localStorage.getItem(I18N_KEY) || 'zh';
  let dict = {};

  function setLang(lang){
    current = lang;
    localStorage.setItem(I18N_KEY, lang);
    return loadDict().then(applyTranslations);
  }

  function loadDict(){
    const url = `/static/js/i18n/${current}.json`;
    return fetch(url).then(r=>r.json()).then(d=>{ dict = d; });
  }

  function t(key){ return dict[key] || key; }

  function applyTranslations(){
    document.querySelectorAll('[data-i18n]').forEach(el=>{
      const key = el.getAttribute('data-i18n');
      const text = t(key);
      if (el.placeholder !== undefined && el.tagName === 'INPUT'){
        el.placeholder = text;
      } else {
        el.textContent = text;
      }
    });
  }

  window.ChanI18n = { setLang, t, init: () => loadDict().then(applyTranslations), current: () => current };
})();


