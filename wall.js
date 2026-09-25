'use strict';

// The introduction opens on one recording, then expands into 24 real clips.
window.createDataWall = function () {
  const wall = document.querySelector('#data-wall');
  const video = document.querySelector('#wall-video');
  const toggle = document.querySelector('#wall-toggle');
  const replay = document.querySelector('#wall-replay');
  const dialogs = document.querySelectorAll('dialog');
  const compact = window.matchMedia('(max-width: 600px)');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let pausedByUser = false;
  let motionOptIn = false;
  const loopStart = 9;
  let preservedTime = reducedMotion.matches ? loopStart : 0;
  let visible = false;
  let pendingPlay = false;

  function allowedToPlay() {
    return visible && !pausedByUser && (!reducedMotion.matches || motionOptIn) && !document.hidden && !document.querySelector('dialog[open]') &&
      !Array.from(document.querySelectorAll('video')).some(other => other !== video && !other.paused && !other.ended);
  }
  function updateControl() {
    const paused = video.paused;
    toggle.classList.toggle('is-paused', paused);
    toggle.setAttribute('aria-label', paused ? 'Play background video' : 'Pause background video');
    toggle.querySelector('span').textContent = paused ? 'Play background video' : 'Pause background video';
  }
  function syncPlayback() {
    if (!allowedToPlay()) {
      video.pause();
    } else if (video.paused && !pendingPlay) {
      pendingPlay = true;
      video.play().catch(error => {
        if (error.name !== 'AbortError' && allowedToPlay()) pausedByUser = true;
      }).finally(() => {
        pendingPlay = false;
        updateControl();
        if (allowedToPlay() && video.paused) syncPlayback();
      });
    }
    updateControl();
  }
  function updateSource() {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) preservedTime = video.currentTime;
    const suffix = compact.matches ? '-mobile' : '';
    const still = reducedMotion.matches && !motionOptIn ? '-expanded' : '';
    video.poster = `assets/images/wall-expand${suffix}${still}.webp`;
    video.src = `assets/videos/wall-expand${suffix}.mp4`;
    video.onloadedmetadata = () => {
      video.currentTime = Math.min(preservedTime, Math.max(0, video.duration - 0.1));
      syncPlayback();
    };
    syncPlayback();
  }
  toggle.addEventListener('click', () => {
    pausedByUser = !video.paused;
    if (!pausedByUser) motionOptIn = true;
    if (!pausedByUser) document.querySelectorAll('video').forEach(other => { if (other !== video) other.pause(); });
    syncPlayback();
  });
  replay.addEventListener('click', () => {
    pausedByUser = false;
    motionOptIn = true;
    preservedTime = 0;
    video.currentTime = 0;
    document.querySelectorAll('video').forEach(other => { if (other !== video) other.pause(); });
    syncPlayback();
  });
  video.addEventListener('ended', () => {
    video.currentTime = loopStart;
    syncPlayback();
  });
  video.addEventListener('play', updateControl);
  video.addEventListener('pause', updateControl);
  new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    syncPlayback();
  }, {threshold:0}).observe(wall);
  document.addEventListener('visibilitychange', syncPlayback);
  // Defer until the foreground player or modal has finished changing state.
  ['play', 'pause', 'ended'].forEach(type => document.addEventListener(type, event => {
    if (event.target instanceof HTMLVideoElement && event.target !== video) queueMicrotask(syncPlayback);
  }, true));
  dialogs.forEach(dialog => dialog.addEventListener('close', syncPlayback));
  reducedMotion.addEventListener('change', () => { motionOptIn = false; syncPlayback(); });
  compact.addEventListener('change', updateSource);
  updateSource();
};
