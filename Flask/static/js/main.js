/* ============================================================
   RESTAURANT RECOMMENDATION SYSTEM — MAIN JS
   ============================================================ */

// ── Navbar scroll behavior ──────────────────────────────────
(function () {
  const navbar = document.querySelector('.navbar');
  if (!navbar) return;

  let lastScroll = 0;

  window.addEventListener('scroll', () => {
    const currentScroll = window.scrollY;

    if (currentScroll > 80) {
      navbar.style.boxShadow = '0 2px 20px rgba(0,0,0,0.3)';
    } else {
      navbar.style.boxShadow = 'none';
    }

    lastScroll = currentScroll;
  });
})();

// ── Mobile menu toggle ──────────────────────────────────────
function toggleMenu() {
  const links = document.querySelector('.nav-links');
  const hamburger = document.querySelector('.hamburger');

  if (!links) return;

  const isOpen = links.style.display === 'flex';

  if (isOpen) {
    links.style.display = 'none';
    hamburger.setAttribute('aria-expanded', 'false');
  } else {
    links.style.cssText = `
      display: flex;
      flex-direction: column;
      position: absolute;
      top: 68px;
      left: 0;
      right: 0;
      background: #1a1410;
      padding: 1rem 2rem;
      gap: 4px;
      border-top: 1px solid rgba(255,255,255,0.1);
      z-index: 999;
    `;
    hamburger.setAttribute('aria-expanded', 'true');
  }
}

// ── Intersection Observer for animations ───────────────────
(function () {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.style.animationPlayState = 'running';
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1 }
  );

  document.querySelectorAll('.step-card, .scenario-card, .stat-item').forEach((el) => {
    el.style.animationPlayState = 'paused';
    observer.observe(el);
  });
})();

// ── Smooth scroll for anchor links ─────────────────────────
document.querySelectorAll('a[href^="#"]').forEach((link) => {
  link.addEventListener('click', (e) => {
    const target = document.querySelector(link.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth' });
    }
  });
});

// ── Close autocomplete on outside click ────────────────────
document.addEventListener('click', (e) => {
  const ac = document.getElementById('autocomplete');
  if (ac && !e.target.closest('.input-group')) {
    ac.style.display = 'none';
  }
});

// ── Animate numbers on scroll (stats) ─────────────────────
(function () {
  const stats = document.querySelectorAll('.stat-num');
  if (!stats.length) return;

  const animateValue = (el, start, end, duration, suffix) => {
    const range = end - start;
    let startTime = null;

    const step = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      const eased = progress < 0.5 ? 2 * progress * progress : -1 + (4 - 2 * progress) * progress;
      el.textContent = Math.floor(start + range * eased).toLocaleString() + suffix;
      if (progress < 1) requestAnimationFrame(step);
    };

    requestAnimationFrame(step);
  };

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const el = entry.target;
          const text = el.textContent;

          if (text.includes('7,000')) animateValue(el, 0, 7000, 1500, '+');
          else if (text.includes('100')) animateValue(el, 0, 100, 1200, '+');
          else if (text.includes('99')) animateValue(el, 0, 99, 1000, '%');

          observer.unobserve(el);
        }
      });
    },
    { threshold: 0.5 }
  );

  stats.forEach((el) => observer.observe(el));
})();

// ── Table row hover highlight ──────────────────────────────
(function () {
  document.querySelectorAll('.result-row').forEach((row) => {
    row.addEventListener('mouseenter', () => {
      row.style.background = 'rgba(200,64,26,0.04)';
    });
    row.addEventListener('mouseleave', () => {
      row.style.background = '';
    });
  });
})();

console.log('%c🍽 Restaurant Recommendation System', 'color: #c8401a; font-size: 16px; font-weight: bold;');
console.log('%cPowered by Content-Based Filtering · TF-IDF · Cosine Similarity', 'color: #6b6158; font-size: 12px;');
