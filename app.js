'use strict';

const samples = window.TOUCHSCALE_SAMPLES;
const samplePreviewCount = 4;
const grid = document.querySelector('#scene-grid');
const dialog = document.querySelector('#scene-dialog');
const search = document.querySelector('#sample-search');
let currentFilter = 'all';
let expanded = false;
let currentSampleIndex = 0;
let dialogSamples = samples;
let dialogView = 'sensors';
const primarySettings = ['Laboratory', 'Kitchen', 'Workbench'];
const videoObserver = new IntersectionObserver(entries => entries.forEach(entry => {
  if (!entry.isIntersecting) entry.target.pause();
}), {threshold: 0});
function watchVideo(video) {
  videoObserver.observe(video);
  video.addEventListener('play', () => {
    document.querySelectorAll('video').forEach(other => { if (other !== video) other.pause(); });
  });
}

function escapeText(text) {
  return String(text).replace(/[&<>"']/g, character => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[character]));
}
function filteredSamples() {
  const query = search.value.trim().toLowerCase();
  return samples.filter(sample => {
    const settingMatches = currentFilter === 'all' || (currentFilter === 'Everyday' ? !primarySettings.includes(sample.category) : sample.category === currentFilter);
    const searchMatches = !query || `${sample.name} ${sample.category} ${sample.label} ${sample.description}`.toLowerCase().includes(query);
    return settingMatches && searchMatches;
  });
}
function showSample(id, open = true) {
  dialogSamples = filteredSamples();
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
  const query = search.value.trim().toLowerCase();
  const filtered = filteredSamples();
  const useLimit = currentFilter === 'all' && !query && !expanded;
  const shown = useLimit ? filtered.slice(0, samplePreviewCount) : filtered;
  grid.querySelectorAll('video').forEach(video => { video.pause(); videoObserver.unobserve(video); video.removeAttribute('src'); video.load(); });
  grid.replaceChildren(...shown.map(sample => {
    const card = document.createElement('article');
    card.className = 'scene-card';
    card.innerHTML = `<div class="scene-video"><video data-video-id="${escapeText(sample.id)}" data-src="${escapeText(sample.video)}" poster="${escapeText(sample.image)}" playsinline muted loop preload="none" tabindex="-1" aria-label="${escapeText(sample.name)} — head RGB recording"></video><button class="sample-play" aria-label="Play ${escapeText(sample.name)}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 11 7-11 7Z"/></svg></button></div><div class="scene-caption"><h3>${escapeText(sample.name)}</h3><button class="sample-expand" aria-label="View all sensors for ${escapeText(sample.name)}">All sensors <span aria-hidden="true">↗</span></button></div>`;
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
  document.querySelector('#explorer-count').textContent = filtered.length === samples.length ? `${shown.length} of ${samples.length} video samples` : `${filtered.length} matching ${filtered.length === 1 ? 'video' : 'videos'}`;
  document.querySelector('#empty-state').hidden = filtered.length !== 0;
  const showAll = document.querySelector('#show-all');
  showAll.hidden = currentFilter !== 'all' || Boolean(query) || filtered.length <= samplePreviewCount;
  showAll.innerHTML = expanded ? 'Show fewer videos <span aria-hidden="true">↑</span>' : `Show all ${samples.length} videos <span aria-hidden="true">↓</span>`;
}
function selectFilter(value) {
  currentFilter = value;
  expanded = false;
  document.querySelectorAll('[data-filter]').forEach(button => {
    const active = button.dataset.filter === value;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  renderSamples();
}
document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => selectFilter(button.dataset.filter)));
search.addEventListener('input', renderSamples);
document.querySelector('#show-all').addEventListener('click', () => { expanded = !expanded; renderSamples(); });
document.querySelector('#reset-search').addEventListener('click', () => { search.value = ''; expanded = false; selectFilter('all'); search.focus(); });
document.querySelectorAll('[data-sample-id]').forEach(button => button.addEventListener('click', () => showSample(button.dataset.sampleId)));
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

const phases = {
  before: { rgb:'cloth-before', touch:'touch-before', depth:'depth-before', left:'wrist-left-before', right:'wrist-right-before', time:'0.7', description:'before contact' },
  contact: { rgb:'cloth-contact', touch:'touch-contact', depth:'depth-contact', left:'wrist-left', right:'wrist-right', time:'6.0', description:'during wringing' },
  after: { rgb:'cloth-after', touch:'touch-after', depth:'depth-after', left:'wrist-left-after', right:'wrist-right-after', time:'8.7', description:'after release' }
};
const modalityNames = {rgb:'Egocentric RGB',touch:'Bimanual tactile contact',depth:'Head depth',left:'Left wrist RGB',right:'Right wrist RGB'};
document.querySelectorAll('[data-phase]').forEach(button => button.addEventListener('click', () => {
  const phase = phases[button.dataset.phase];
  for (const key of Object.keys(modalityNames)) {
    const image = document.querySelector(`#observation-${key}`);
    image.src = `assets/images/${phase[key]}.webp`;
    image.alt = `${modalityNames[key]} ${phase.description}, at ${phase.time} seconds`;
  }
  document.querySelector('#snapshot-time').textContent = `${phase.time} s`;
  document.querySelectorAll('[data-phase]').forEach(item => {
    item.classList.toggle('active', item === button);
    item.setAttribute('aria-pressed', String(item === button));
  });
}));

const chartData = {
  scenes:[['Laboratory',104.3],['Kitchen',88.2],['Workbench',98.7],['Medical / First Aid',39.8],['Office',62.6],['Packing / Shipping',48.4],['Bedroom',39.7],['Teleop Alignment',3.8],['Active Tactile',14.5]],
  verbs:[['Place',170],['Pour',94],['Transfer',87],['Put',78],['Lift',65],['Insert',63],['Wipe',56],['Press',54],['Fold',51],['Open',51],['Pull',51],['Close',45]]
};
const sceneChart = document.querySelector('#scene-composition');
const scenePie = window.createScenePie(sceneChart, chartData.scenes);
function renderChart(kind) {
  const bars = document.querySelector('#distribution-bars');
  scenePie.reset();
  sceneChart.hidden = kind !== 'scenes';
  bars.hidden = kind === 'scenes';
  document.querySelector('#chart-unit').textContent = kind === 'scenes' ? '500 h estimate' : '1,964 descriptions · 500 h estimate';
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
