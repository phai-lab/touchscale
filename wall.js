'use strict';

// One recording expands into twenty tiles, followed by the centered wordmark.
window.createDataWall = function () {
  const wall = document.querySelector('#data-wall');
  const video = document.querySelector('#wall-video');
  const toggle = document.querySelector('#wall-toggle');
  const wordmark = document.querySelector('.wall-wordmark');
  const dialogs = document.querySelectorAll('dialog');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let pausedByUser = false;
  let motionOptIn = false;
  const loopStart = 8;
  let preservedTime = reducedMotion.matches ? loopStart : 0;
  let visible = false;
  let pendingPlay = false;

  function syncWordmark() {
    const progress = reducedMotion.matches && !motionOptIn ? 1 : Math.max(0, Math.min(1, (video.currentTime - 6.5) / 1.5));
    wordmark.style.setProperty('--wordmark-opacity', progress.toFixed(3));
  }

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
    syncWordmark();
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
    const suffix = '';
    const still = reducedMotion.matches && !motionOptIn ? '-expanded' : '';
    // Version the film and loop point together for returning visitors.
    video.poster = `assets/images/wall-expand${suffix}${still}.webp?v=wall20-hd-1`;
    video.src = `assets/videos/wall-expand${suffix}.mp4?v=wall20-hd-1`;
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
  video.addEventListener('ended', () => {
    video.currentTime = loopStart;
    syncPlayback();
  });
  video.addEventListener('play', updateControl);
  video.addEventListener('pause', updateControl);
  ['timeupdate', 'seeking', 'seeked', 'loadeddata'].forEach(type => video.addEventListener(type, syncWordmark));
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
  updateSource();
};
