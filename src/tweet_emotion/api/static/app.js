/* tweet-emotion-pipeline demo page. Vanilla ES5, no build step.

   Every number on the page arrives from /version, /presets, and /terms. Until a value lands
   its slot keeps the [todo] chip, and when the page is opened from a file or the API is down
   nothing is invented. */
(function () {
  'use strict';

  var API_ONLINE = location.protocol === 'http:' || location.protocol === 'https:';
  var LOCAL_BASE = 'http://127.0.0.1:8000';
  /* Fallback only; /version replaces it with params.api.max_chars. */
  var DEFAULT_MAX_CHARS = 1000;
  var REDUCED = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  var MODEL_NAMES = {
    logreg: 'Logistic regression (logreg)',
    nb: 'Multinomial naive Bayes (nb)',
    xgboost: 'Gradient boosted trees (xgboost)'
  };

  /* Helpers */

  function $(id) {
    return document.getElementById(id);
  }

  function empty(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function chip() {
    return el('span', 'todo', '[todo]');
  }

  function setText(id, text) {
    var node = $(id);
    if (node) node.textContent = text;
  }

  function setChip(id) {
    var node = $(id);
    if (!node) return;
    empty(node);
    node.appendChild(chip());
  }

  function isNum(v) {
    return typeof v === 'number' && isFinite(v);
  }

  function fmt(v, dp) {
    return isNum(v) ? v.toFixed(dp) : null;
  }

  function fmtPct(v) {
    return isNum(v) ? (v * 100).toFixed(2) + '%' : null;
  }

  function fmtThreshold(v) {
    return isNum(v) ? String(Number(v.toFixed(4))) : null;
  }

  function fmtInt(v) {
    return isNum(v) ? v.toLocaleString('en-US') : null;
  }

  function fmtDate(iso) {
    if (!iso) return null;
    var d = new Date(iso);
    if (isNaN(d.getTime())) return null;
    var m = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return m[d.getUTCMonth()] + ' ' + d.getUTCDate() + ', ' + d.getUTCFullYear();
  }

  function lowerFirst(s) {
    return s ? s.charAt(0).toLowerCase() + s.slice(1) : '';
  }

  function getJSON(path) {
    return fetch(path, { headers: { Accept: 'application/json' } }).then(function (r) {
      if (!r.ok) throw new Error(path + ' returned ' + r.status);
      return r.json();
    });
  }

  /* Receipts */

  var TILES = ['accuracy', 'roc_auc', 'f1', 'trained', 'latency'];

  function markUnmeasured(key) {
    var tile = document.querySelector('[data-tile="' + key + '"]');
    if (!tile) return;
    var k = tile.querySelector('.kicker');
    if (k) k.textContent = 'Unmeasured';
    setChip('slot-' + key);
  }

  function fillTile(key, text) {
    if (text === null || text === undefined) {
      markUnmeasured(key);
      return;
    }
    setText('slot-' + key, text);
  }

  function allUnmeasured() {
    TILES.forEach(markUnmeasured);
  }

  /* Character limit and counter */

  var textarea = $('tweet');
  var counter = $('counter');
  var maxChars = DEFAULT_MAX_CHARS;
  var disclaimer = '';

  function updateCounter() {
    var n = textarea.value.length;
    counter.textContent = fmtInt(n) + ' / ' + fmtInt(maxChars);
    if (n >= maxChars) counter.classList.add('over'); else counter.classList.remove('over');
  }

  function setMaxChars(n) {
    if (isNum(n) && n > 0) maxChars = Math.floor(n);
    textarea.setAttribute('maxlength', String(maxChars));
    updateCounter();
  }

  function loadVersion() {
    return getJSON('/version').then(function (v) {
      var m = v.metrics || {};
      fillTile('accuracy', fmt(m.accuracy, 4));
      fillTile('roc_auc', fmt(m.roc_auc, 4));
      fillTile('f1', fmt(m.f1, 4));
      fillTile('trained', fmtDate(v.trained_at || m.trained_at));
      var lt = v.loadtest;
      if (lt && isNum(lt.p95_ms)) {
        fillTile('latency', fmt(lt.p95_ms, 2) + ' ms');
        setText('slot-host', 'POST /predict on ' + (lt.host || 'an unrecorded host'));
      } else {
        markUnmeasured('latency');
      }
      if (v.model_version) setText('slot-version', 'v' + v.model_version);
      if (v.disclaimer) {
        disclaimer = v.disclaimer;
        setText('slot-disclaimer', v.disclaimer);
      }
      var nTest = fmtInt(m.n_test);
      if (nTest !== null) setText('slot-ntest', nTest);
      var t = fmtThreshold(m.threshold);
      if (t !== null) setText('slot-threshold', t);
      var p = v.params || {};
      if (isNum(p.max_chars)) setMaxChars(p.max_chars);
    }).catch(allUnmeasured);
  }

  /* Presets */

  var presets = [];
  var selected = -1;

  function setPressed(index) {
    Array.prototype.forEach.call($('presets').querySelectorAll('button'), function (b, i) {
      b.setAttribute('aria-pressed', i === index ? 'true' : 'false');
    });
  }

  function describePreset(p) {
    var desc = $('preset-desc');
    empty(desc);
    if (!p) return;
    desc.appendChild(document.createTextNode((p.description || '') + ' '));
    var bits = [];
    if (p.label) bits.push('true label ' + p.label);
    if (isNum(p.probability_happiness)) bits.push('p(happiness) ' + p.probability_happiness.toFixed(4));
    if (p.tweet_id !== undefined && p.tweet_id !== null) bits.push('tweet id ' + p.tweet_id);
    if (bits.length) desc.appendChild(el('span', 'mono', bits.join(', ') + '.'));
  }

  function applyPreset(index) {
    var p = presets[index];
    if (!p) return;
    textarea.value = p.text || '';
    updateCounter();
    clearInvalid();
    selected = index;
    setPressed(index);
    describePreset(p);
  }

  /* Editing the text away from the chosen preset drops the selection, since the
     description and true label no longer apply. */
  function syncSelection() {
    if (selected < 0) return;
    var p = presets[selected];
    if (p && textarea.value === (p.text || '')) return;
    selected = -1;
    setPressed(-1);
    describePreset(null);
  }

  function buildPresetButtons() {
    var box = $('presets');
    empty(box);
    presets.forEach(function (p, i) {
      var b = el('button', 'btn', p.title || ('Preset ' + (i + 1)));
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', function () {
        applyPreset(i);
        textarea.focus();
      });
      box.appendChild(b);
    });
  }

  function shellQuote(s) {
    return "'" + String(s).replace(/'/g, "'\\''") + "'";
  }

  function buildCurl(text) {
    var base = API_ONLINE ? location.origin : LOCAL_BASE;
    var body = JSON.stringify({ text: text });
    var pre = $('code-curl');
    empty(pre);
    pre.appendChild(el('code', null,
      'curl -X POST ' + base + '/predict -H "Content-Type: application/json" -d ' + shellQuote(body)));
  }

  function presetsUnavailable() {
    var box = $('presets');
    empty(box);
    var s = el('span', 'mono');
    s.appendChild(document.createTextNode('Presets need the API: '));
    s.appendChild(chip());
    box.appendChild(s);
  }

  function loadPresets() {
    return getJSON('/presets').then(function (data) {
      presets = Array.isArray(data.presets) ? data.presets : [];
      if (!presets.length) {
        presetsUnavailable();
        return;
      }
      buildPresetButtons();
      applyPreset(0);
      buildCurl(presets[0].text || '');
    }).catch(presetsUnavailable);
  }

  /* Result card */

  var form = $('scorer');
  var firstResult = true;

  function describedBy(node) {
    return (node.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean);
  }

  function clearInvalid() {
    textarea.removeAttribute('aria-invalid');
    var ids = describedBy(textarea).filter(function (id) { return id.indexOf('err-') !== 0; });
    textarea.setAttribute('aria-describedby', ids.join(' '));
  }

  /* Below this width the result panel stacks under the form instead of sitting beside it. */
  function stacked() {
    return window.innerWidth <= 1000;
  }

  function showState(which) {
    $('result-empty').hidden = which !== 'empty';
    $('result-error').hidden = which !== 'error';
    var body = $('result-body');
    body.hidden = which !== 'result';
    if (which === 'result' && firstResult) {
      firstResult = false;
      /* Reading offsetWidth forces a layout with the element visible and still at opacity 0,
         so the transition to .in runs even in a background tab where animation frames pause.
         Under reduced motion the stylesheet pins .fade to full opacity and no transition. */
      void body.offsetWidth;
      body.classList.add('in');
    }
    if (which !== 'empty' && stacked()) {
      document.querySelector('.result').scrollIntoView({
        behavior: REDUCED ? 'auto' : 'smooth',
        block: 'start'
      });
    }
  }

  function showError(title, text, items) {
    setText('result-error-title', title);
    setText('result-error-text', text || '');
    var list = $('result-error-list');
    empty(list);
    (items || []).forEach(function (item) {
      var li = el('li');
      if (typeof item === 'string') {
        li.textContent = item;
      } else {
        if (item.id) li.id = item.id;
        li.textContent = item.text;
      }
      list.appendChild(li);
    });
    list.hidden = !(items && items.length);
    showState('error');
  }

  /* FastAPI 422 bodies: {detail: [{loc, msg, type}]}. Pydantic prefixes custom validator
     messages with "Value error, ", which is dropped so the sentence reads plainly. */
  function showValidation(detail) {
    clearInvalid();
    var items = [];
    var seen = {};
    if (Array.isArray(detail)) {
      detail.forEach(function (d) {
        var loc = Array.isArray(d.loc) ? d.loc.filter(function (x) { return x !== 'body'; }) : [];
        var name = loc.length ? String(loc[loc.length - 1]) : 'request';
        var msg = lowerFirst(String(d.msg || 'invalid value').replace(/^Value error, /, ''));
        var base = 'err-' + name.replace(/[^A-Za-z0-9_-]/g, '_');
        seen[base] = (seen[base] || 0) + 1;
        var id = seen[base] === 1 ? base : base + '-' + seen[base];
        items.push({ id: id, text: name + ': ' + msg });
      });
    } else if (detail) {
      items.push(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    var ids = items.filter(function (it) { return it.id; }).map(function (it) { return it.id; });
    if (ids.length) {
      textarea.setAttribute('aria-invalid', 'true');
      textarea.setAttribute('aria-describedby', describedBy(textarea).concat(ids).join(' '));
    }
    showError('The API rejected the request', 'Fix the text and submit again.', items);
    /* On a stacked layout the panel was just scrolled into view, so focus lands on the error
       heading; moving it to the textarea would scroll the error back off screen. */
    if (stacked()) {
      $('result-error-title').focus({ preventScroll: true });
    } else {
      textarea.focus();
    }
  }

  function renderSplit(ph) {
    var box = $('split');
    if (!isNum(ph)) {
      box.hidden = true;
      return;
    }
    var sad = Math.min(1, Math.max(0, 1 - ph));
    $('split-sad').style.width = (sad * 100).toFixed(2) + '%';
    setText('split-sad-l', 'sadness ' + fmtPct(sad));
    setText('split-happy-l', 'happiness ' + fmtPct(ph));
    box.hidden = false;
  }

  function renderWhy(terms) {
    var list = $('why-list');
    var note = $('why-empty');
    var none = $('why-none');
    empty(list);
    /* null: the classifier has no per-term weights. []: it does, but no vocabulary term
       survived normalisation, so only the intercept scored the text. */
    if (!Array.isArray(terms)) {
      list.hidden = true;
      note.hidden = false;
      none.hidden = true;
      return;
    }
    if (!terms.length) {
      list.hidden = true;
      note.hidden = true;
      none.hidden = false;
      return;
    }
    var max = 0;
    terms.forEach(function (t) {
      if (isNum(t.contribution) && Math.abs(t.contribution) > max) max = Math.abs(t.contribution);
    });
    terms.forEach(function (t) {
      var c = isNum(t.contribution) ? t.contribution : 0;
      var li = el('li', 'why-row');
      li.appendChild(el('span', 'why-term', String(t.term || '')));
      var bar = el('span', 'why-bar');
      bar.setAttribute('aria-hidden', 'true');
      var fill = el('span', 'why-fill ' + (c >= 0 ? 'pos' : 'neg'));
      /* Half the track per side, so the largest bar reaches the edge. */
      fill.style.width = (max > 0 ? (Math.abs(c) / max) * 50 : 0).toFixed(1) + '%';
      bar.appendChild(fill);
      li.appendChild(bar);
      li.appendChild(el('span', 'why-v', (c > 0 ? '+' : '') + c.toFixed(4)));
      list.appendChild(li);
    });
    list.hidden = false;
    note.hidden = true;
    none.hidden = true;
  }

  /* When the scored text is still a preset, say whether the model matched its crowd label. */
  function renderTruth(data) {
    var line = $('r-truth');
    var p = selected >= 0 ? presets[selected] : null;
    if (!p || !p.label || textarea.value !== (p.text || '')) {
      line.hidden = true;
      return;
    }
    line.textContent = 'True label ' + p.label + (p.label === data.label ? ', the model agrees.' : ', the model missed it.');
    line.hidden = false;
  }

  function showResult(data) {
    clearInvalid();
    var prob = $('r-prob');
    var pct = fmtPct(data.probability);
    if (pct === null) {
      empty(prob);
      prob.appendChild(chip());
    } else {
      prob.textContent = pct;
    }
    setText('r-label', data.label || '');
    renderSplit(data.probability_happiness);
    renderWhy(data.terms);
    renderTruth(data);
    var t = fmtThreshold(data.threshold);
    if (t === null) setChip('r-threshold'); else setText('r-threshold', t);
    setText('r-model', MODEL_NAMES[data.model] || data.model || '');
    if (data.model_version) setText('r-version', data.model_version); else setChip('r-version');
    var norm = $('r-normalized');
    empty(norm);
    if (typeof data.normalized_text === 'string' && data.normalized_text.length) {
      norm.textContent = data.normalized_text;
    } else {
      norm.appendChild(el('span', 'empty', 'empty after normalisation'));
    }
    setText('r-disclaimer', data.disclaimer || disclaimer);
    showState('result');
  }

  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    if (!API_ONLINE) {
      showError(
        'No API behind this page',
        'Open it through the service (python -m tweet_emotion.api, or the docker run line below) to score a tweet.'
      );
      return;
    }
    var btn = $('submit');
    var label = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Scoring';
    fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ text: textarea.value })
    }).then(function (r) {
      if (r.status === 422) {
        return r.json().then(function (d) { showValidation(d.detail); });
      }
      if (r.status === 413) {
        showError('The API returned status 413', 'The request body is larger than the service accepts. Shorten the text and try again.');
        return;
      }
      if (!r.ok) {
        showError('The API returned status ' + r.status, 'Check the service logs and try again.');
        return;
      }
      return r.json().then(showResult);
    }).catch(function () {
      showError('Could not reach the API', 'The request did not complete. Check that the service is running and try again.');
    }).then(function () {
      btn.disabled = false;
      btn.textContent = label;
    });
  });

  textarea.addEventListener('input', function () {
    updateCounter();
    syncSelection();
  });

  /* Top terms */

  function renderTermList(id, rows, max) {
    var list = $(id);
    empty(list);
    rows.forEach(function (row) {
      var w = isNum(row.weight) ? Math.abs(row.weight) : 0;
      var li = el('li', 'term-row');
      li.appendChild(el('span', 'term-name', String(row.term || '')));
      var bar = el('span', 'bar');
      bar.setAttribute('aria-hidden', 'true');
      var fill = el('span');
      fill.style.width = (max > 0 ? (w / max) * 100 : 0).toFixed(1) + '%';
      bar.appendChild(fill);
      li.appendChild(bar);
      li.appendChild(el('span', 'term-v', isNum(row.weight) ? row.weight.toFixed(4) : ''));
      list.appendChild(li);
    });
  }

  function termsUnavailable(method) {
    $('terms-grid').hidden = true;
    $('terms-empty').hidden = false;
    if (method) {
      setText('slot-terms-method',
        'This model does not expose signed term weights; the file holds ' + method + ' instead.');
    }
  }

  function loadTerms() {
    return getJSON('/terms').then(function (data) {
      var happy = data && Array.isArray(data.happiness) ? data.happiness : [];
      var sad = data && Array.isArray(data.sadness) ? data.sadness : [];
      if (!data || data.available !== true || !happy.length || !sad.length) {
        termsUnavailable(data && data.method);
        return;
      }
      var max = 0;
      happy.concat(sad).forEach(function (row) {
        if (isNum(row.weight) && Math.abs(row.weight) > max) max = Math.abs(row.weight);
      });
      renderTermList('terms-happy-list', happy, max);
      renderTermList('terms-sad-list', sad, max);
      $('terms-grid').hidden = false;
      $('terms-empty').hidden = true;
      var n = Math.max(happy.length, sad.length);
      setText('slot-terms-method', 'Method: ' + (data.method || 'model weights') + ', top ' + n + ' terms per side.');
    }).catch(function () {
      termsUnavailable();
    });
  }

  /* Copy buttons */

  function legacyCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try {
      ok = document.execCommand('copy');
    } catch (e) {
      ok = false;
    }
    document.body.removeChild(ta);
    return ok;
  }

  var copyStatus = $('copy-status');

  /* The button reads Copied for a moment; the live region says what was copied, built from
     the button's aria-label ("Copy the curl command" becomes "Copied the curl command"). */
  function flash(btn) {
    var label = btn.textContent;
    var what = (btn.getAttribute('aria-label') || '').replace(/^Copy /, '');
    btn.textContent = 'Copied';
    if (copyStatus) copyStatus.textContent = 'Copied ' + (what || 'the command');
    setTimeout(function () {
      btn.textContent = label;
      if (copyStatus) copyStatus.textContent = '';
    }, 1500);
  }

  Array.prototype.forEach.call(document.querySelectorAll('.btn-copy'), function (btn) {
    btn.addEventListener('click', function () {
      var pre = $(btn.getAttribute('data-copy'));
      if (!pre) return;
      var text = pre.textContent;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          flash(btn);
        }).catch(function () {
          if (legacyCopy(text)) flash(btn);
        });
      } else if (legacyCopy(text)) {
        flash(btn);
      }
    });
  });

  /* Boot */

  setMaxChars(DEFAULT_MAX_CHARS);
  if (API_ONLINE) {
    loadVersion();
    loadPresets();
    loadTerms();
  } else {
    allUnmeasured();
    presetsUnavailable();
    termsUnavailable();
  }
})();
