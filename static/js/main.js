// ===== MINDWAVE MAIN JS =====

// Mobile nav toggle
function toggleNav() {
  document.body.classList.toggle('nav-open');
}

// Close nav when clicking outside
document.addEventListener('click', (e) => {
  if (!e.target.closest('.nav')) {
    document.body.classList.remove('nav-open');
  }
});

// Scroll reveal animation
function initScrollReveal() {
  const els = document.querySelectorAll('.scroll-reveal');
  if (!els.length) return;
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add('visible'), i * 80);
      }
    });
  }, { threshold: 0.1 });
  els.forEach(el => obs.observe(el));
}

// Animate progress/mood bars on load
function animateBars() {
  document.querySelectorAll('.pb-fill, .mood-fill, .ars-fill, .ed-fill, .ci-fill').forEach(el => {
    const target = el.style.width;
    el.style.width = '0%';
    setTimeout(() => { el.style.width = target; }, 300);
  });
}

// Wellness score counter animation
function animateCounter(el, target, duration = 1200) {
  if (!el) return;
  let start = 0;
  const step = (timestamp) => {
    if (!start) start = timestamp;
    const progress = Math.min((timestamp - start) / duration, 1);
    el.textContent = Math.floor(progress * target);
    if (progress < 1) requestAnimationFrame(step);
    else el.textContent = target;
  };
  requestAnimationFrame(step);
}

// Highlight active nav link
function setActiveNav() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-links a').forEach(a => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });
}

// Char counter for journal textarea
function initCharCounter() {
  const ta = document.getElementById('journalText');
  const cc = document.getElementById('charCount');
  if (!ta || !cc) return;
  ta.addEventListener('input', () => {
    const len = ta.value.length;
    cc.textContent = `${len} character${len !== 1 ? 's' : ''}`;
    cc.style.color = len < 10 ? 'var(--danger)' : 'var(--text3)';
  });
}

// Slider value display (for check-in page)
function initSliders() {
  document.querySelectorAll('input[type="range"]').forEach(slider => {
    const targetId = slider.id.replace('Slider', 'Val');
    const target = document.getElementById(targetId);
    if (target) {
      slider.addEventListener('input', () => {
        target.textContent = slider.value;
      });
    }
  });
}

// Flash message auto-dismiss
function initFlash() {
  document.querySelectorAll('.success-banner, .auth-error').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity 0.5s ease';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 500);
    }, 5000);
  });
}

// Smooth scroll for anchor links
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
}

// Wellness score stat card animation
function initStatCounters() {
  const wellnessEl = document.getElementById('wellnessNum');
  if (wellnessEl) {
    const val = parseInt(wellnessEl.textContent) || 0;
    animateCounter(wellnessEl, val);
  }
}

// Typing indicator loop (for hero phone mockup)
function initTypingIndicator() {
  const indicator = document.querySelector('.typing-indicator');
  if (!indicator) return;
  // Already animated via CSS, just ensure it loops
}

// Dashboard: refresh data every 60 seconds if on dashboard page
function initDashboardRefresh() {
  if (!document.querySelector('.dash-page')) return;
  // Charts are loaded once on page load via loadCharts()
  // Can add auto-refresh here if needed
}

// Form validation
function initFormValidation() {
  const forms = document.querySelectorAll('form');
  forms.forEach(form => {
    form.addEventListener('submit', e => {
      const required = form.querySelectorAll('[required]');
      let valid = true;
      required.forEach(input => {
        if (!input.value.trim()) {
          valid = false;
          input.style.borderColor = 'var(--danger)';
          input.addEventListener('input', () => {
            input.style.borderColor = '';
          }, { once: true });
        }
      });
      if (!valid) {
        e.preventDefault();
        // Scroll to first invalid field
        const firstInvalid = form.querySelector('[required]:invalid, [style*="danger"]');
        if (firstInvalid) firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
  });

  // Password confirm match
  const pw = document.getElementById('password');
  const confirm = document.getElementById('confirm');
  if (pw && confirm) {
    confirm.addEventListener('input', () => {
      if (confirm.value && confirm.value !== pw.value) {
        confirm.style.borderColor = 'var(--danger)';
      } else {
        confirm.style.borderColor = pw.value === confirm.value && confirm.value ? 'var(--accent3)' : '';
      }
    });
  }
}

// Ripple effect on buttons
function initRipple() {
  document.querySelectorAll('.btn-primary, .btn-ghost').forEach(btn => {
    btn.addEventListener('click', function(e) {
      const rect = this.getBoundingClientRect();
      const ripple = document.createElement('span');
      ripple.style.cssText = `
        position:absolute;width:6px;height:6px;border-radius:50%;
        background:rgba(255,255,255,0.4);pointer-events:none;
        left:${e.clientX - rect.left - 3}px;top:${e.clientY - rect.top - 3}px;
        transform:scale(0);animation:ripple 0.5s ease forwards;z-index:1;
      `;
      if (!this.style.position || this.style.position === 'static') {
        this.style.position = 'relative';
        this.style.overflow = 'hidden';
      }
      this.appendChild(ripple);
      setTimeout(() => ripple.remove(), 500);
    });
  });

  // Inject ripple keyframes
  if (!document.getElementById('rippleStyle')) {
    const style = document.createElement('style');
    style.id = 'rippleStyle';
    style.textContent = '@keyframes ripple { to { transform: scale(30); opacity: 0; } }';
    document.head.appendChild(style);
  }
}

// Tab switching (history page)
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const content = document.getElementById('tab-' + name);
  if (content) content.style.display = 'block';
  if (event && event.target) event.target.classList.add('active');
}

// Nav scroll effect
function initNavScroll() {
  const nav = document.getElementById('mainNav');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    if (window.scrollY > 20) {
      nav.style.background = 'rgba(8,12,20,0.97)';
    } else {
      nav.style.background = 'rgba(8,12,20,0.9)';
    }
  }, { passive: true });
}

// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => {
  initScrollReveal();
  animateBars();
  initCharCounter();
  initSliders();
  initFlash();
  initSmoothScroll();
  initStatCounters();
  initTypingIndicator();
  initDashboardRefresh();
  initFormValidation();
  initRipple();
  initNavScroll();
  setActiveNav();
});
