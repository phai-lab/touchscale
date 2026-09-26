'use strict';

const samples = window.TOUCHSCALE_SAMPLES;
const grid = document.querySelector('#scene-grid');
const dialog = document.querySelector('#scene-dialog');
let currentSampleIndex = 0;
let dialogSamples = samples;
let dialogView = 'sensors';
const videoObserver = new IntersectionObserver(entries => entries.forEach(entry => {
  if (!entry.isIntersecting) entry.target.pause();
}), {threshold: 0});
function watchVideo(video) {
  videoObserver.observe(video);
  video.addEventListener('play', () => {
    document.querySelectorAll('video').forEach(other => {
      const sameGroup = video.dataset.playbackGroup && video.dataset.playbackGroup === other.dataset.playbackGroup;
      if (other !== video && !sameGroup) other.pause();
    });
  });
}

// UMI-style opening: both views play together while visible. A foreground
// sample, video wall, or dialog still takes precedence over the pair.
const openingVideos = [...document.querySelectorAll('[data-playback-group="opening"]')];
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let openingVisible = false;
function playOpening() {
  if (!openingVisible || document.hidden || reduceMotion.matches || document.querySelector('dialog[open]')) return;
  if ([...document.querySelectorAll('video')].some(video => !openingVideos.includes(video) && video.id !== 'wall-video' && !video.paused)) return;
  document.querySelector('#wall-video').pause();
  openingVideos.forEach(video => video.play().catch(() => {}));
}
openingVideos.forEach(video => video.addEventListener('play', () => {
  document.querySelectorAll('video').forEach(other => {
    if (!openingVideos.includes(other)) other.pause();
  });
}));
new IntersectionObserver(entries => {
  openingVisible = entries[0].isIntersecting;
  if (openingVisible) playOpening();
  else openingVideos.forEach(video => video.pause());
}, {threshold:.15}).observe(document.querySelector('.opening-pair'));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) openingVideos.forEach(video => video.pause());
  else playOpening();
});

function escapeText(text) {
  return String(text).replace(/[&<>"']/g, character => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[character]));
}
function showSample(id, open = true) {
  dialogSamples = samples;
  dialogView = 'sensors';
  currentSampleIndex = dialogSamples.findIndex(sample => sample.id === String(id));
  if (currentSampleIndex < 0) return;
  const sample = dialogSamples[currentSampleIndex];
  const inlineVideo = document.querySelector(`[data-video-id="${sample.id}"]`);
  const resumeTime = open && inlineVideo ? inlineVideo.currentTime : 0;
  showRecording(sample, open, resumeTime);
}
function showRecording(sample, open, resumeTime = 0) {
  const video = document.querySelector('#dialog-video');
  document.querySelectorAll('video').forEach(item => item.pause());
  video.poster = sample.poster;
  video.src = dialogView === 'sensors' ? sample.sensors : sample.video;
  video.setAttribute('aria-label', `${sample.name}, ${dialogView === 'sensors' ? 'synchronized multimodal recording' : 'egocentric RGB recording'}`);
  document.querySelector('#dialog-title').textContent = sample.name;
  document.querySelector('#dialog-view-label').textContent = dialogView === 'sensors' ? 'Touch on fixed hand templates' : 'Egocentric RGB';
  if (open) {
    dialog.showModal();
    document.body.style.overflow = 'hidden';
  }
  video.currentTime = resumeTime;
  video.play().catch(() => {});
}
function nextSample(direction) {
  currentSampleIndex = (currentSampleIndex + direction + dialogSamples.length) % dialogSamples.length;
  showRecording(dialogSamples[currentSampleIndex], false);
}
function renderSamples() {
  grid.querySelectorAll('video').forEach(video => { video.pause(); videoObserver.unobserve(video); video.removeAttribute('src'); video.load(); });
  grid.replaceChildren(...samples.map(sample => {
    const card = document.createElement('article');
    card.className = 'media-column half scene-card';
    card.innerHTML = `<div class="scene-video"><video data-video-id="${escapeText(sample.id)}" data-src="${escapeText(sample.video)}" poster="${escapeText(sample.image)}" playsinline muted loop preload="none" tabindex="-1" aria-label="${escapeText(sample.name)} — head RGB recording"></video><button class="sample-play" aria-label="Play ${escapeText(sample.name)}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 11 7-11 7Z"/></svg></button></div><div class="scene-caption"><h4>${escapeText(sample.name)}</h4><button class="sample-expand" aria-label="View all sensors for ${escapeText(sample.name)}">All sensors <span aria-hidden="true">↗</span></button></div>`;
    const video = card.querySelector('video');
    const play = card.querySelector('.sample-play');
    watchVideo(video);
    play.addEventListener('click', () => {
      if (!video.getAttribute('src')) video.src = video.dataset.src;
      video.controls = true;
      video.tabIndex = 0;
      video.play().catch(() => { video.controls = true; });
    });
    video.addEventListener('play', () => { if (document.activeElement === play) video.focus({preventScroll:true}); play.hidden = true; });
    card.querySelector('.sample-expand').addEventListener('click', () => showSample(sample.id));
    return card;
  }));
}
document.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
document.querySelector('#previous-sample').addEventListener('click', () => nextSample(-1));
document.querySelector('#next-sample').addEventListener('click', () => nextSample(1));
dialog.addEventListener('close', () => {
  const video = document.querySelector('#dialog-video');
  video.pause();
  video.removeAttribute('src');
  video.load();
  document.body.style.overflow = '';
});
dialog.addEventListener('click', event => {
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
});
dialog.addEventListener('keydown', event => {
  if (event.target.tagName !== 'VIDEO' && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
    event.preventDefault();
    nextSample(event.key === 'ArrowLeft' ? -1 : 1);
  }
});
renderSamples();

const chartData = {
  // data_diversity.pdf: retain the listed projected hours (sum 503.6),
  // rather than rescaling them to the figure's nominal 500-hour headline.
  scenes:[['Workbench',128.6],['Laboratory',92.9],['Kitchen',85.7],['Office',55.2],['Packing / Shipping',45.7],['Medical / First Aid',41.9],['Bedroom',35.1],['Active Tactile',14.7],['Teleop Alignment',3.8]],
  verbs:[['Place',170],['Pour',94],['Transfer',87],['Put',78],['Lift',65],['Insert',63],['Wipe',56],['Press',54],['Fold',51],['Open',51],['Pull',51],['Close',45]]
};
const sceneChart = document.querySelector('#scene-composition');
const scenePie = window.createScenePie(sceneChart, chartData.scenes);
function renderChart(kind) {
  const bars = document.querySelector('#distribution-bars');
  scenePie.reset();
  sceneChart.hidden = kind !== 'scenes';
  bars.hidden = kind === 'scenes';
  document.querySelector('#chart-unit').textContent = kind === 'scenes' ? '~500 h' : '1,964 descriptions · ~500 h';
  if (kind === 'scenes') return;
  const maximum = Math.max(...chartData.verbs.map(row => row[1]));
  bars.setAttribute('aria-label', 'Verb frequencies across an estimated 1,964 task descriptions for 500 hours');
  bars.replaceChildren(...chartData.verbs.map(([name, value]) => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `<span>${name}</span><div class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:${value / maximum * 100}%"></div></div><strong>${value}</strong>`;
    row.title = `${name}: ${value} task descriptions`;
    return row;
  }));
}
document.querySelectorAll('[data-chart]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('[data-chart]').forEach(item => { item.classList.toggle('active', item === button); item.setAttribute('aria-pressed', String(item === button)); });
  renderChart(button.dataset.chart);
}));
renderChart('scenes');

document.querySelectorAll('.robot-card video, #dialog-video').forEach(watchVideo);

// Native video controls keep the task rows visually aligned with UMI.
// The shared playback group permits either or both views to play independently.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) document.querySelectorAll('.robot-card video').forEach(video => video.pause());
});

const figureDialog = document.querySelector('#figure-dialog');
const figureImage = document.querySelector('#figure-image');
document.querySelectorAll('.figure-open').forEach(button => {
  button.addEventListener('click', () => {
    const source = button.querySelector('img');
    document.querySelector('#figure-title').textContent = button.dataset.figureTitle;
    figureImage.src = source.src;
    figureImage.alt = source.alt;
    // Fit tall setup diagrams; let wide, detailed figures scroll on small screens.
    figureImage.classList.toggle('is-wide', Number(source.getAttribute('width')) > Number(source.getAttribute('height')));
    document.querySelectorAll('video').forEach(video => video.pause());
    figureDialog.showModal();
    document.body.style.overflow = 'hidden';
    document.querySelector('.figure-viewport').scrollTo(0, 0);
  });
});
document.querySelector('#figure-close').addEventListener('click', () => figureDialog.close());
figureDialog.addEventListener('close', () => { document.body.style.overflow = ''; });
figureDialog.addEventListener('click', event => {
  const bounds = figureDialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) figureDialog.close();
});

window.createDataWall();
