window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
  var revealTargets = document.querySelectorAll('.section, .brand-hero, .footer');
  revealTargets.forEach(function(target) {
    target.classList.add('reveal');
  });

  function flushRevealsInView() {
    var vh = window.innerHeight || document.documentElement.clientHeight;
    document.querySelectorAll('.reveal:not(.is-visible)').forEach(function(el) {
      var r = el.getBoundingClientRect();
      if (r.bottom > 0 && r.top < vh * 0.98) {
        el.classList.add('is-visible');
      }
    });
  }

  if ('IntersectionObserver' in window) {
    var revealObserver = new IntersectionObserver(function(entries, observer) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0,
      rootMargin: '0px 0px 12% 0px'
    });

    revealTargets.forEach(function(target) {
      revealObserver.observe(target);
    });
  } else {
    revealTargets.forEach(function(target) {
      target.classList.add('is-visible');
    });
  }

  requestAnimationFrame(function() {
    flushRevealsInView();
  });

  window.addEventListener('load', function() {
    document.body.classList.add('is-loaded');
    document.body.classList.remove('loading');
    flushRevealsInView();
  });

  window.addEventListener('hashchange', flushRevealsInView);

  var backToTop = $('.back-to-top');
  $(window).on('scroll', function() {
    if ($(this).scrollTop() > 300) backToTop.fadeIn();
    else backToTop.fadeOut();
  });

  var bibtexButton = document.getElementById('bibtex-copy-button');
  var bibtexCode = document.getElementById('bibtex-code');
  var bibtexStatus = document.getElementById('bibtex-copy-status');
  if (bibtexButton && bibtexCode && bibtexStatus) {
    bibtexButton.addEventListener('click', async function() {
      try {
        await navigator.clipboard.writeText(bibtexCode.innerText.trim());
        bibtexStatus.textContent = 'Copied';
        setTimeout(function() {
          bibtexStatus.textContent = '';
        }, 1500);
      } catch (err) {
        bibtexStatus.textContent = 'Copy failed';
      }
    });
  }
});
