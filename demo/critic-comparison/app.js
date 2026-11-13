(() => {
  const cases = window.DEMO_DATA.cases;
  let activeKey = 'success';
  let videos = [];
  let duration = 0;
  let seeking = false;

  const $ = (id) => document.getElementById(id);
  const stage = $('videoStage');
  const scrubber = $('scrubber');
  const formatTime = (seconds) => {
    const value = Math.max(0, seconds || 0);
    return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(Math.floor(value % 60)).padStart(2, '0')}.${String(Math.floor((value % 1) * 100)).padStart(2, '0')}`;
  };
  const pretty = (value) => (value || 'unknown').replaceAll('_', ' ');
  const latestRow = (rows, time) => {
    let found = null;
    for (const row of rows) {
      if (row.source_time_s <= time + .01) found = row;
      else break;
    }
    return found;
  };

  function loadCase(key) {
    activeKey = key;
    const item = cases[key];
    document.querySelectorAll('.case-tab').forEach((button) => button.classList.toggle('active', button.dataset.case === key));
    stage.className = `video-stage ${item.videos.length === 2 ? 'two' : 'three'}`;
    stage.replaceChildren();
    videos = item.videos.map((video, index) => {
      const tile = document.createElement('div');
      tile.className = 'camera-tile';
      const element = document.createElement('video');
      element.src = video.src;
      if (video.rotate === 180) element.classList.add('rotate-180');
      element.preload = 'metadata';
      element.muted = true;
      element.playsInline = true;
      if (index === 0) {
        element.addEventListener('timeupdate', syncUI);
        element.addEventListener('loadedmetadata', () => {
          duration = element.duration;
          $('duration').textContent = formatTime(duration);
          renderMarks(item);
          syncUI();
        });
        element.addEventListener('ended', () => $('playButton').textContent = '▶');
      }
      const label = document.createElement('span');
      label.className = 'camera-label';
      label.textContent = video.label;
      tile.append(element, label);
      stage.append(tile);
      return element;
    });
    $('annotatedLink').href = item.annotated_video;
    $('predictionsLink').href = item.predictions;
    $('summaryLink').href = item.summary;
    $('storyTitle').textContent = item.story.title;
    $('storyBody').textContent = item.story.body;
    $('eventStrip').innerHTML = item.events.map((event) => `<article class="event ${event.kind || ''}"><time>${event.time}</time><b>${event.title}</b><p>${event.detail}</p></article>`).join('');
    $('playButton').textContent = '▶';
    updateTelemetry(null, item);
  }

  function renderMarks(item) {
    $('assessmentMarks').innerHTML = item.prediction_rows.map((row) => {
      const left = (row.source_time_s / duration) * 100;
      const fail = row.safety.failure_accepted ? 'fail' : '';
      return `<i class="${fail}" style="left:${left}%"></i>`;
    }).join('');
  }

  function updateTelemetry(row, item) {
    const card = $('verdictCard');
    card.className = 'verdict-card';
    if (!row) {
      $('verdictLabel').textContent = 'WARMING UP';
      $('verdictTitle').textContent = 'Awaiting first window';
      $('verdictDetail').textContent = 'The critic uses 8 chronological frames per camera.';
      ['phaseState', 'pipetteState', 'tubeState', 'assessmentIndex', 'latency'].forEach((id) => $(id).textContent = '—');
      $('phaseConfidence').textContent = '—';
      $('confirmationText').textContent = '0 / 2';
      $('action').textContent = 'CONTINUE';
      document.querySelectorAll('.confirmation-track i').forEach((bar) => bar.classList.remove('on'));
      return;
    }
    const failed = row.safety.failure_accepted;
    card.classList.add(failed ? 'fail' : 'safe');
    $('verdictLabel').textContent = failed ? 'FAILURE ACCEPTED' : 'STATE CONSISTENT';
    $('verdictTitle').textContent = failed ? 'Stop robot motion' : 'No safety failure';
    $('verdictDetail').textContent = failed ? pretty(row.safety.reason || 'unsafe object support persisted') : 'Observed supports remain compatible with the active phase.';
    $('phaseState').textContent = pretty(row.phase.status);
    $('phaseConfidence').textContent = `${Math.round((row.phase.confidence || 0) * 100)}% CONF.`;
    $('pipetteState').textContent = pretty(row.safety.pipette_support);
    $('tubeState').textContent = pretty(row.safety.tube_support);
    const confirmations = row.safety.failure_confirmation_count || 0;
    $('confirmationText').textContent = `${confirmations} / 2`;
    document.querySelectorAll('.confirmation-track i').forEach((bar, index) => bar.classList.toggle('on', index < confirmations));
    $('assessmentIndex').textContent = `#${String(row.assessment_index + 1).padStart(2, '0')}`;
    const phaseMs = row.phase.metrics?.agent_latency_ms || 0;
    $('latency').textContent = `${Math.round(phaseMs + (item.safety_mean_ms || 0))} ms`;
    $('action').textContent = failed ? 'STOP' : 'CONTINUE';
  }

  function syncUI() {
    if (!videos.length) return;
    const master = videos[0];
    const item = cases[activeKey];
    videos.slice(1).forEach((video) => {
      if (!seeking && Math.abs(video.currentTime - master.currentTime) > .12) video.currentTime = master.currentTime;
    });
    if (!seeking && duration) scrubber.value = Math.round((master.currentTime / duration) * 1000);
    $('currentTime').textContent = formatTime(master.currentTime);
    updateTelemetry(latestRow(item.prediction_rows, master.currentTime), item);
  }

  $('playButton').addEventListener('click', () => {
    if (!videos.length) return;
    if (videos[0].paused) {
      if (videos[0].ended) videos.forEach((video) => video.currentTime = 0);
      Promise.all(videos.map((video) => video.play())).catch(() => {});
      $('playButton').textContent = 'Ⅱ';
    } else {
      videos.forEach((video) => video.pause());
      $('playButton').textContent = '▶';
    }
  });
  scrubber.addEventListener('input', () => {
    seeking = true;
    const time = (Number(scrubber.value) / 1000) * duration;
    videos.forEach((video) => video.currentTime = time);
    $('currentTime').textContent = formatTime(time);
    updateTelemetry(latestRow(cases[activeKey].prediction_rows, time), cases[activeKey]);
  });
  scrubber.addEventListener('change', () => seeking = false);
  $('muteButton').addEventListener('click', () => {
    const muted = !videos[0].muted;
    videos.forEach((video) => video.muted = muted);
    $('muteButton').textContent = muted ? 'MUTE' : 'SOUND';
  });
  document.querySelectorAll('.case-tab').forEach((button) => button.addEventListener('click', () => loadCase(button.dataset.case)));
  document.addEventListener('keydown', (event) => {
    if (event.code === 'Space' && event.target.tagName !== 'INPUT') { event.preventDefault(); $('playButton').click(); }
  });
  loadCase(activeKey);
})();
