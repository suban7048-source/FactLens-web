/**
 * app.js — Fake News Detection System Frontend
 * ============================================
 * Handles:
 *  - Theme toggle (dark/light, persisted in localStorage)
 *  - API calls to Flask backend (/api/predict, /api/metrics)
 *  - Result card rendering with animated confidence ring
 *  - Chart.js model performance charts
 *  - Loading states and error handling
 *  - Intersection Observer reveal animations
 *  - Mobile navigation
 *  - Sample text insertion
 */

'use strict';

/* ── Constants ───────────────────────────────────────────────── */
const API_BASE = window.location.origin; // same-origin Flask serving
const ENDPOINTS = {
  predict: `${API_BASE}/api/predict`,
  metrics: `${API_BASE}/api/metrics`,
  health:  `${API_BASE}/api/health`,
};

const SAMPLE_TEXTS = {
  fake: `BREAKING: Scientists EXPOSED! Secret government documents leaked online reveal that top NASA officials have been hiding evidence of alien contact for decades. Whistleblowers inside the deep state confirm the shocking cover-up that mainstream media refuses to report. Share this before it gets deleted! The globalist elite are furious that this information is getting out. You won't believe what they've been hiding from the American public all these years.`,

  real: `The Federal Reserve announced on Wednesday that it would maintain its benchmark interest rate at its current level, citing ongoing concerns about inflation and labor market conditions. Federal Reserve Chair Jerome Powell said in a press conference that policymakers would continue to monitor economic data before making any adjustments to monetary policy. According to the Bureau of Labor Statistics, the unemployment rate held steady at 3.8 percent in August, while the consumer price index rose 3.2 percent year-over-year, slightly above the Fed's 2 percent target.`,
};

/* ── DOM References ──────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const els = {
  body:           document.body,
  themeToggle:    $('theme-toggle'),
  hamburger:      $('hamburger'),
  navLinks:       $('nav-links'),
  navbar:         $('navbar'),
  newsInput:      $('news-input'),
  charCounter:    $('char-counter'),
  detectBtn:      $('detect-btn'),
  resetBtn:       $('reset-btn'),
  loadingState:   $('loading-state'),
  resultCard:     $('result-card'),
  placeholder:    $('result-placeholder'),
  errorToast:     $('error-toast'),
  errorMsg:       $('error-msg'),
  resultBadge:    $('result-badge'),
  confidencePct:  $('confidence-pct'),
  ringFill:       $('ring-fill'),
  barInner:       $('bar-inner'),
  resultExplain:  $('result-explanation'),
  modelName:      $('model-name'),
  apiStatusBar:   $('api-status-bar'),
  apiStatusText:  $('api-status-text'),
  metricsTable:   $('metrics-table-body'),
  bestModelBadge: $('best-model-name'),
  cmCells:        null, // populated later
};

/* ── Theme Management ────────────────────────────────────────── */
function initTheme() {
  const saved = localStorage.getItem('fnd-theme') || 'dark';
  applyTheme(saved);
}

function applyTheme(theme) {
  els.body.setAttribute('data-theme', theme);
  if (els.themeToggle) {
    els.themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
    els.themeToggle.setAttribute('aria-label',
      theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
  }
  localStorage.setItem('fnd-theme', theme);
}

function toggleTheme() {
  const current = els.body.getAttribute('data-theme');
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

/* ── Mobile Nav ──────────────────────────────────────────────── */
function initNav() {
  if (els.hamburger && els.navLinks) {
    els.hamburger.addEventListener('click', () => {
      els.navLinks.classList.toggle('open');
    });

    // Close on link click
    els.navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => els.navLinks.classList.remove('open'));
    });
  }

  // Navbar scroll shadow
  window.addEventListener('scroll', () => {
    if (els.navbar) {
      els.navbar.classList.toggle('scrolled', window.scrollY > 20);
    }
  }, { passive: true });
}

/* ── Character Counter ───────────────────────────────────────── */
function initCharCounter() {
  if (!els.newsInput || !els.charCounter) return;

  els.newsInput.addEventListener('input', () => {
    const len = els.newsInput.value.length;
    els.charCounter.textContent = `${len.toLocaleString()} / 50,000 chars`;
    els.charCounter.className = 'char-counter' + (len > 45000 ? ' warn' : '');
  });
}

/* ── Sample Text Insertion ───────────────────────────────────── */
function initSampleButtons() {
  document.querySelectorAll('[data-sample]').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.getAttribute('data-sample');
      if (SAMPLE_TEXTS[key] && els.newsInput) {
        els.newsInput.value = SAMPLE_TEXTS[key];
        els.newsInput.dispatchEvent(new Event('input'));
        els.newsInput.focus();
      }
    });
  });
}

/* ── API Health Check ────────────────────────────────────────── */
async function checkHealth() {
  if (!els.apiStatusBar || !els.apiStatusText) return;
  try {
    const resp = await fetch(ENDPOINTS.health, { signal: AbortSignal.timeout(4000) });
    const data = await resp.json();
    if (data.status === 'ok' && data.model_ready) {
      els.apiStatusBar.classList.remove('offline');
      els.apiStatusText.textContent = `API Online — Model: ${data.model}`;
    } else {
      els.apiStatusBar.classList.add('offline');
      els.apiStatusText.textContent = 'Model not ready — run train_model.py';
    }
  } catch {
    els.apiStatusBar.classList.add('offline');
    els.apiStatusText.textContent = 'API Offline — start the Flask server';
  }
}

/* ── Result UI Helpers ───────────────────────────────────────── */
function showPlaceholder() {
  if (els.placeholder)  els.placeholder.classList.remove('hidden');
  if (els.loadingState) els.loadingState.classList.remove('active');
  if (els.resultCard) {
    els.resultCard.classList.remove('visible');
    els.resultCard.style.display = 'none';
  }
  if (els.errorToast)   els.errorToast.classList.remove('visible');
}

function showLoading() {
  if (els.placeholder)  els.placeholder.classList.add('hidden');
  if (els.loadingState) els.loadingState.classList.add('active');
  if (els.resultCard) {
    els.resultCard.classList.remove('visible');
    els.resultCard.style.display = 'none';
  }
  if (els.errorToast)   els.errorToast.classList.remove('visible');
}

function showError(msg) {
  if (els.loadingState) els.loadingState.classList.remove('active');
  if (els.placeholder)  els.placeholder.classList.remove('hidden');
  if (els.errorToast && els.errorMsg) {
    els.errorMsg.textContent = msg;
    els.errorToast.classList.add('visible');
    setTimeout(() => els.errorToast.classList.remove('visible'), 6000);
  }
}

function animateRing(pct, labelClass) {
  if (!els.ringFill) return;
  // r=39, circumference = 2*PI*39 ≈ 245
  const circumference = 245;
  const offset = circumference - (circumference * pct / 100);

  // Remove previous classes
  els.ringFill.className = `ring-fill ${labelClass}`;
  els.barInner.className = `confidence-bar-inner ${labelClass}`;

  // Reset then animate
  els.ringFill.style.strokeDashoffset = circumference;
  els.barInner.style.width = '0%';

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      els.ringFill.style.strokeDashoffset = offset;
      els.barInner.style.width = `${pct}%`;
    });
  });
}

function showResult(data) {
  if (els.loadingState) els.loadingState.classList.remove('active');
  if (els.placeholder)  els.placeholder.classList.add('hidden');

  const isReal      = data.label === 'REAL';
  const labelClass  = isReal ? 'real' : 'fake';
  const pct         = Math.round(data.confidence * 100);
  const icon        = isReal ? '✅' : '🚫';
  const label       = isReal ? 'Real News' : 'Fake News';

  // Badge
  if (els.resultBadge) {
    els.resultBadge.textContent = `${icon} ${label}`;
    els.resultBadge.className = `result-label-badge ${labelClass}`;
  }

  // Confidence text
  if (els.confidencePct) {
    els.confidencePct.textContent = `${pct}%`;
    els.confidencePct.style.color = isReal
      ? 'var(--success)' : 'var(--danger)';
  }

  // Ring + bar animation
  animateRing(pct, labelClass);

  // Explanation
  if (els.resultExplain) {
    els.resultExplain.textContent = data.explanation;
  }

  // Model name
  if (els.modelName) {
    els.modelName.textContent = data.model;
  }

  // Show card
  if (els.resultCard) {
    els.resultCard.style.display = 'flex';
    // Force reflow for animation
    void els.resultCard.offsetHeight;
    els.resultCard.classList.add('visible');
  }

  // Scroll result into view on mobile
  if (window.innerWidth < 900 && els.resultCard) {
    setTimeout(() => {
      els.resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 200);
  }
}

/* ── Main Predict Flow ───────────────────────────────────────── */
async function handleDetect() {
  const text = els.newsInput ? els.newsInput.value.trim() : '';

  if (!text) {
    showError('Please paste a news article or headline before detecting.');
    return;
  }
  if (text.length < 15) {
    showError('Text is too short. Please provide at least 15 characters.');
    return;
  }

  showLoading();
  if (els.detectBtn) els.detectBtn.disabled = true;

  try {
    const resp = await fetch(ENDPOINTS.predict, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(30000),
    });

    const data = await resp.json();

    if (!resp.ok) {
      showError(data.error || `Server error (${resp.status}). Try again.`);
      return;
    }

    showResult(data);

  } catch (err) {
    if (err.name === 'TimeoutError') {
      showError('Request timed out. Make sure the Flask server is running.');
    } else if (err.name === 'TypeError') {
      showError('Cannot connect to the API. Start the Flask server: python app.py');
    } else {
      showError(`Unexpected error: ${err.message}`);
    }
  } finally {
    if (els.detectBtn) els.detectBtn.disabled = false;
  }
}

function handleReset() {
  if (els.newsInput) {
    els.newsInput.value = '';
    els.newsInput.dispatchEvent(new Event('input'));
    els.newsInput.focus();
  }
  showPlaceholder();
}

/* ── Metrics & Charts ────────────────────────────────────────── */
let perfChart = null;
let cmChart   = null;

async function loadMetrics() {
  try {
    const resp = await fetch(ENDPOINTS.metrics, {
      signal: AbortSignal.timeout(8000),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    renderMetrics(data);
  } catch {
    // Silently fail — metrics are supplementary
  }
}

function renderMetrics(data) {
  const models   = data.models || {};
  const bestName = data.best_model || '';
  const modelNames = Object.keys(models);

  if (!modelNames.length) return;

  // Update best model badge in header
  if (els.bestModelBadge) {
    els.bestModelBadge.textContent = bestName;
  }

  // Update top stat cards
  const bestMetrics = models[bestName] || models[modelNames[0]];
  if (bestMetrics) {
    setStatCard('stat-accuracy',  bestMetrics.accuracy);
    setStatCard('stat-precision', bestMetrics.precision);
    setStatCard('stat-recall',    bestMetrics.recall);
    setStatCard('stat-f1',        bestMetrics.f1_score);
  }

  // Populate table
  if (els.metricsTable) {
    els.metricsTable.innerHTML = modelNames.map(name => {
      const m      = models[name];
      const isBest = name === bestName;
      return `
        <tr class="${isBest ? 'best-row' : ''}">
          <td class="model-name-cell">
            ${name}
            ${isBest ? '<span class="best-badge">🏆 Best</span>' : ''}
          </td>
          <td>${pct(m.accuracy)}</td>
          <td>${pct(m.precision)}</td>
          <td>${pct(m.recall)}</td>
          <td>${pct(m.f1_score)}</td>
          <td>${m.train_time_sec ?? '—'}s</td>
        </tr>
      `;
    }).join('');
  }

  // Performance chart (bar — accuracy / F1 comparison)
  const perfCanvas = document.getElementById('perf-chart');
  if (perfCanvas && window.Chart) {
    if (perfChart) perfChart.destroy();
    const isDark = document.body.getAttribute('data-theme') !== 'light';
    const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
    const textColor = isDark ? '#94a3b8' : '#475569';

    perfChart = new Chart(perfCanvas, {
      type: 'bar',
      data: {
        labels: modelNames,
        datasets: [
          {
            label: 'Accuracy',
            data: modelNames.map(n => models[n].accuracy),
            backgroundColor: 'rgba(99,102,241,0.75)',
            borderColor: 'rgba(99,102,241,1)',
            borderWidth: 1.5,
            borderRadius: 6,
          },
          {
            label: 'F1-Score',
            data: modelNames.map(n => models[n].f1_score),
            backgroundColor: 'rgba(139,92,246,0.65)',
            borderColor: 'rgba(139,92,246,1)',
            borderWidth: 1.5,
            borderRadius: 6,
          },
          {
            label: 'Precision',
            data: modelNames.map(n => models[n].precision),
            backgroundColor: 'rgba(6,182,212,0.55)',
            borderColor: 'rgba(6,182,212,1)',
            borderWidth: 1.5,
            borderRadius: 6,
          },
          {
            label: 'Recall',
            data: modelNames.map(n => models[n].recall),
            backgroundColor: 'rgba(16,185,129,0.55)',
            borderColor: 'rgba(16,185,129,1)',
            borderWidth: 1.5,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: textColor, boxWidth: 12, font: { size: 12 } },
          },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(2)}%`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: textColor, font: { size: 11 } },
          },
          y: {
            min: 0.7,
            max: 1.0,
            grid: { color: gridColor },
            ticks: {
              color: textColor,
              font: { size: 11 },
              callback: v => `${(v * 100).toFixed(0)}%`,
            },
          },
        },
      },
    });
  }

  // Confusion Matrix for best model
  const cm = bestMetrics?.confusion_matrix;
  if (cm && cm.length === 2) {
    // cm[0][0]=TN, cm[0][1]=FP, cm[1][0]=FN, cm[1][1]=TP
    const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
    setCMCell('cm-tp', tp, 'True Positive');
    setCMCell('cm-fp', fp, 'False Positive');
    setCMCell('cm-fn', fn, 'False Negative');
    setCMCell('cm-tn', tn, 'True Negative');
  }

  // Dataset info
  const dsInfo = $('dataset-info');
  if (dsInfo && data.dataset_size) {
    dsInfo.textContent =
      `Trained on ${data.dataset_size.toLocaleString()} samples · ` +
      `${data.train_size?.toLocaleString() ?? '—'} train · ` +
      `${data.test_size?.toLocaleString() ?? '—'} test`;
  }
}

function pct(v) {
  return v != null ? `${(v * 100).toFixed(2)}%` : '—';
}

function setStatCard(id, val) {
  const el = $(id);
  if (el && val != null) {
    el.textContent = `${(val * 100).toFixed(1)}%`;
  }
}

function setCMCell(id, val, label) {
  const el = $(id);
  if (el) {
    const num  = el.querySelector('.cm-num');
    if (num) num.textContent = val?.toLocaleString() ?? '—';
  }
}

/* ── Intersection Observer (reveal animations) ───────────────── */
function initReveal() {
  const observer = new IntersectionObserver(
    entries => entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    }),
    { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
  );

  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}

/* ── Confidence demo bars in hero (cosmetic animation) ───────── */
function initHeroDemoBars() {
  const fills = document.querySelectorAll('.demo-confidence-fill');
  setTimeout(() => {
    fills.forEach(f => {
      const target = f.classList.contains('real') ? '94%' : '88%';
      f.style.width = target;
    });
  }, 800);
}

/* ── Theme-aware chart re-render ─────────────────────────────── */
function watchThemeForCharts() {
  const observer = new MutationObserver(() => {
    // If metrics already loaded, rebuild charts with new colors
    if (perfChart) {
      const btn = $('reload-metrics-btn');
      loadMetrics();
    }
  });
  observer.observe(document.body, {
    attributes: true,
    attributeFilter: ['data-theme'],
  });
}

/* ── Keyboard Shortcuts ──────────────────────────────────────── */
function initKeyboard() {
  document.addEventListener('keydown', e => {
    // Ctrl+Enter or Cmd+Enter to detect
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleDetect();
    }
    // Escape to reset
    if (e.key === 'Escape' && document.activeElement === els.newsInput) {
      handleReset();
    }
  });
}

/* ── Bootstrap ───────────────────────────────────────────────── */
function init() {
  initTheme();
  initNav();
  initCharCounter();
  initSampleButtons();
  initReveal();
  initHeroDemoBars();
  initKeyboard();
  watchThemeForCharts();

  // Event listeners
  if (els.themeToggle) els.themeToggle.addEventListener('click', toggleTheme);
  if (els.detectBtn)   els.detectBtn.addEventListener('click', handleDetect);
  if (els.resetBtn)    els.resetBtn.addEventListener('click', handleReset);

  // Show placeholder on load
  showPlaceholder();

  // Check API health (non-blocking)
  checkHealth();

  // Load metrics when performance section scrolls into view
  const perfSection = $('performance');
  if (perfSection) {
    const metricObserver = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting) {
        loadMetrics();
        metricObserver.disconnect();
      }
    }, { threshold: 0.1 });
    metricObserver.observe(perfSection);
  }
}

// Run after DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
