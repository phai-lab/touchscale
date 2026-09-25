/* Scene composition: a local, accessible SVG chart with no dependencies. */
(function () {
  'use strict';

  const svgNamespace = 'http://www.w3.org/2000/svg';
  const palette = ['#ad5754', '#d4a066', '#7b9d96', '#a78fac', '#7a94b1', '#bd9a82', '#98ae86', '#aaadb4', '#d1b4c2'];
  let chartCount = 0;

  window.createScenePie = function (container, sceneRows) {
    if (!container) return null;
    const rows = sceneRows.filter(row => Array.isArray(row) && typeof row[0] === 'string' && Number.isFinite(row[1]) && row[1] > 0);
    const total = rows.reduce((sum, row) => sum + row[1], 0);
    if (!rows.length || !total) return null;

    const chart = document.createElement('div');
    chart.className = 'scene-pie';
    chart.setAttribute('role', 'group');
    chart.setAttribute('aria-label', `Estimated scene composition, ${total.toLocaleString('en-US')} hours in total`);

    const legend = document.createElement('ul');
    legend.className = 'scene-pie-legend';
    legend.setAttribute('aria-label', 'Scene categories and hours');

    const stage = document.createElement('div');
    stage.className = 'scene-pie-stage';
    const svg = document.createElementNS(svgNamespace, 'svg');
    svg.classList.add('scene-pie-svg');
    svg.setAttribute('viewBox', '0 0 400 400');
    svg.setAttribute('role', 'group');
    svg.setAttribute('aria-label', 'Hours by scene category');

    const tooltip = document.createElement('div');
    tooltip.className = 'scene-pie-tooltip';
    tooltip.id = `scene-pie-tooltip-${++chartCount}`;
    tooltip.setAttribute('role', 'tooltip');
    tooltip.hidden = true;
    const tooltipName = document.createElement('strong');
    const tooltipValue = document.createElement('span');
    tooltip.append(tooltipName, tooltipValue);

    const slices = [];
    const buttons = [];
    const midpoints = [];
    const removeListeners = [];
    let hoverIndex = -1;
    let keyboardIndex = -1;
    let touchIndex = -1;
    let activeIndex = -1;
    let startAngle = -Math.PI / 2;

    function listen(element, name, callback) {
      element.addEventListener(name, callback);
      removeListeners.push(() => element.removeEventListener(name, callback));
    }

    function point(angle, radius) {
      return [200 + radius * Math.cos(angle), 200 + radius * Math.sin(angle)];
    }

    function positionTooltip() {
      if (activeIndex < 0 || tooltip.hidden) return;
      const width = stage.clientWidth;
      const height = stage.clientHeight;
      if (!width || !height) return;
      const angle = midpoints[activeIndex];
      const [x, y] = point(angle, 106).map((coordinate, index) => coordinate / 400 * (index ? height : width));
      const left = Math.cos(angle) >= 0 ? x + 13 : x - tooltip.offsetWidth - 13;
      tooltip.style.left = `${Math.max(8, Math.min(left, width - tooltip.offsetWidth - 8))}px`;
      tooltip.style.top = `${Math.max(8, Math.min(y - tooltip.offsetHeight / 2, height - tooltip.offsetHeight - 8))}px`;
    }

    function renderActive() {
      activeIndex = hoverIndex >= 0 ? hoverIndex : keyboardIndex >= 0 ? keyboardIndex : touchIndex;
      chart.classList.toggle('has-active', activeIndex >= 0);
      slices.forEach((slice, index) => {
        const active = index === activeIndex;
        slice.classList.toggle('is-active', active);
        buttons[index].classList.toggle('is-active', active);
        slice.setAttribute('aria-pressed', String(index === touchIndex));
        buttons[index].setAttribute('aria-pressed', String(index === touchIndex));
      });
      tooltip.hidden = activeIndex < 0;
      if (activeIndex >= 0) {
        const [name, hours] = rows[activeIndex];
        tooltipName.textContent = name;
        tooltipValue.textContent = `${hours.toFixed(1)} h · ${(hours / total * 100).toFixed(2)}%`;
        tooltip.style.setProperty('--slice-color', palette[activeIndex % palette.length]);
        positionTooltip();
      }
    }

    function reset() {
      hoverIndex = keyboardIndex = touchIndex = -1;
      renderActive();
    }

    function makeInteractive(element, index, collection) {
      listen(element, 'pointerenter', event => {
        if (event.pointerType === 'touch') return;
        hoverIndex = index;
        renderActive();
      });
      listen(element, 'pointerleave', event => {
        if (event.pointerType === 'touch') return;
        if (hoverIndex === index) hoverIndex = -1;
        renderActive();
      });
      listen(element, 'focus', () => {
        if (!element.matches(':focus-visible')) return;
        keyboardIndex = index;
        renderActive();
      });
      listen(element, 'blur', () => {
        if (keyboardIndex === index) keyboardIndex = -1;
        renderActive();
      });
      listen(element, 'click', event => {
        // A click retains a selection for touch and assistive input; mouse hover is transient.
        if (event.pointerType === 'mouse') return;
        touchIndex = touchIndex === index ? -1 : index;
        renderActive();
      });
      listen(element, 'keydown', event => {
        let next;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % rows.length;
        if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index - 1 + rows.length) % rows.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = rows.length - 1;
        if (next !== undefined) {
          event.preventDefault();
          keyboardIndex = next;
          collection[next].focus({preventScroll: true});
          renderActive();
        } else if (event.key === 'Escape') {
          event.preventDefault();
          reset();
        } else if ((event.key === 'Enter' || event.key === ' ') && element === slices[index]) {
          event.preventDefault();
          touchIndex = touchIndex === index ? -1 : index;
          renderActive();
        }
      });
    }

    rows.forEach(([name, hours], index) => {
      const endAngle = startAngle + hours / total * Math.PI * 2;
      const middle = (startAngle + endAngle) / 2;
      const start = point(startAngle, 174);
      const end = point(endAngle, 174);
      const slice = document.createElementNS(svgNamespace, 'path');
      slice.classList.add('scene-pie-slice');
      slice.setAttribute('d', rows.length === 1
        ? 'M 200 26 A 174 174 0 1 1 200 374 A 174 174 0 1 1 200 26 Z'
        : `M 200 200 L ${start.join(' ')} A 174 174 0 ${endAngle - startAngle > Math.PI ? 1 : 0} 1 ${end.join(' ')} Z`);
      slice.setAttribute('fill', palette[index % palette.length]);
      slice.setAttribute('tabindex', '0');
      slice.setAttribute('role', 'button');
      slice.setAttribute('aria-label', `${name}: ${hours.toFixed(1)} hours, ${(hours / total * 100).toFixed(2)} percent`);
      slice.setAttribute('aria-pressed', 'false');
      slice.style.setProperty('--slice-x', `${Math.cos(middle) * 6}px`);
      slice.style.setProperty('--slice-y', `${Math.sin(middle) * 6}px`);
      slices.push(slice);
      midpoints.push(middle);
      svg.append(slice);

      const item = document.createElement('li');
      const button = document.createElement('button');
      button.className = 'scene-pie-key';
      button.type = 'button';
      button.setAttribute('aria-label', slice.getAttribute('aria-label'));
      button.setAttribute('aria-pressed', 'false');
      const swatch = document.createElement('span');
      swatch.className = 'scene-pie-swatch';
      swatch.setAttribute('aria-hidden', 'true');
      swatch.style.backgroundColor = palette[index % palette.length];
      const label = document.createElement('span');
      label.className = 'scene-pie-label';
      label.textContent = name;
      const value = document.createElement('span');
      value.className = 'scene-pie-value';
      value.textContent = `${hours.toFixed(1)} h`;
      button.append(swatch, label, value);
      item.append(button);
      legend.append(item);
      buttons.push(button);
      makeInteractive(slice, index, slices);
      makeInteractive(button, index, buttons);
      startAngle = endAngle;
    });

    stage.append(svg, tooltip);
    chart.append(legend, stage);
    container.replaceChildren(chart);
    listen(document, 'pointerdown', event => {
      if (!chart.contains(event.target) && touchIndex >= 0) {
        touchIndex = -1;
        renderActive();
      }
    });
    const resizeObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(positionTooltip) : null;
    if (resizeObserver) resizeObserver.observe(stage);
    else listen(window, 'resize', positionTooltip);

    return {
      reset,
      destroy() {
        removeListeners.forEach(remove => remove());
        if (resizeObserver) resizeObserver.disconnect();
        chart.remove();
      }
    };
  };
})();
