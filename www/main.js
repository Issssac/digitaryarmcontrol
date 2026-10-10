import init, { App } from './pkg/arm_sim.js';
import { CytonSerial } from './serial.js';
import { CytonSimulator } from './simulator.js';

const NUM_CH = 8;
const KEYS = [['q', 'a'], ['w', 's'], ['e', 'd'], ['r', 'f']];
// Electrode site of each channel (see the electrode guide) and the movement to
// *intend* during calibration: tense as if starting it, with the arm held still.
const MOVES = [
  { muscle: 'Pronator teres', short: 'Palm down', cue: 'Turn your palm down', how: 'as if starting to pour out a cup' },
  { muscle: 'Brachioradialis', short: 'Hammer curl', cue: 'Hammer curl', how: 'thumb up, as if starting to bend the elbow' },
  { muscle: 'Front of shoulder', short: 'Arm forward', cue: 'Lift your arm forward', how: 'as if raising it in front of you' },
  { muscle: 'Back of shoulder', short: 'Elbow back', cue: 'Pull your elbow back', how: 'as if driving the elbow backwards' },
  { muscle: 'Biceps', short: 'Curl up', cue: 'Curl your forearm up', how: 'palm up, as if lifting a cup' },
  { muscle: 'Triceps', short: 'Press down', cue: 'Press your forearm down', how: 'as if straightening the elbow into the table' },
  { muscle: 'Wrist flexors', short: 'Wrist in', cue: 'Curl your wrist in', how: 'palm side toward the forearm' },
  { muscle: 'Wrist extensors', short: 'Wrist back', cue: 'Bend your wrist back', how: 'knuckles up, as if revving a motorbike' },
];
const SIM_LEVEL = 0.7;
// Simulated tension during a calibration pulse (lands below a real max, like a light intention).
const SIM_PULSE = 0.5;
const UI_INTERVAL_MS = 66;

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v >= 0 ? '+' : '') + v.toFixed(d);

function fatal(message) {
  const el = $('error');
  el.textContent = message;
  el.hidden = false;
}

let app;
try {
  await init();
  app = new App($('gl'));
} catch (err) {
  fatal(`Could not start the renderer: ${err}`);
  throw err;
}

const names = app.joint_names();

// ---- view: sizing, orbit, zoom ---------------------------------------------

const canvas = $('gl');
new ResizeObserver(() => {
  const dpr = Math.min(devicePixelRatio, 2);
  app.resize(Math.round(canvas.clientWidth * dpr), Math.round(canvas.clientHeight * dpr));
}).observe(canvas);

let dragging = false;
canvas.addEventListener('pointerdown', (e) => {
  dragging = true;
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointerup', () => { dragging = false; });
canvas.addEventListener('pointercancel', () => { dragging = false; });
canvas.addEventListener('pointermove', (e) => {
  if (dragging) app.orbit(e.movementX, e.movementY);
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  app.zoom(e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY);
}, { passive: false });

// ---- control source --------------------------------------------------------

let source = 'manual';

function setSource(src) {
  source = src;
  app.set_use_emg(src === 'emg');
  for (const b of $('source').querySelectorAll('button')) {
    b.classList.toggle('active', b.dataset.src === src);
  }
  $('source-hint').textContent = src === 'emg'
    ? 'Arm follows muscle activation. With Simulate on, Q/A · W/S · E/D · R/F fake contractions (Shift = full).'
    : 'Keys Q/A · W/S · E/D · R/F, or drag the sliders (they spring back to 0).';
}

$('source').addEventListener('click', (e) => {
  const src = e.target.dataset?.src;
  if (src) setSource(src);
});

// ---- keyboard --------------------------------------------------------------

const held = new Set();
let shift = false;
const ALL_KEYS = KEYS.flat();

addEventListener('keydown', (e) => {
  shift = e.shiftKey;
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  if (ALL_KEYS.includes(k)) {
    held.add(k);
    e.preventDefault();
  }
});
addEventListener('keyup', (e) => {
  shift = e.shiftKey;
  held.delete(e.key.toLowerCase());
});
addEventListener('blur', () => held.clear());

const keyAxis = (j) => (held.has(KEYS[j][0]) ? 1 : 0) - (held.has(KEYS[j][1]) ? 1 : 0);

// ---- joints panel ----------------------------------------------------------

const joints = names.map((name, j) => {
  const row = document.createElement('div');
  row.className = 'joint';
  row.innerHTML = `
    <div class="name">${name}<small>${KEYS[j][0].toUpperCase()} / ${KEYS[j][1].toUpperCase()}</small></div>
    <input type="range" min="-1" max="1" step="0.01" value="0" aria-label="${name} velocity">
    <div class="angle mono"></div>
    <div class="meter"><div></div></div>`;
  $('joints').append(row);
  return {
    slider: row.querySelector('input'),
    angle: row.querySelector('.angle'),
    meter: row.querySelector('.meter div'),
  };
});

// sliders are velocity commands and spring back on release
addEventListener('pointerup', () => {
  for (const j of joints) j.slider.value = '0';
});

$('reset').onclick = () => app.reset();
$('clear-trail').onclick = () => app.clear_trail();

// ---- mapping ---------------------------------------------------------------

function readMapping() {
  const m = app.mapping();
  return names.map((_, j) => [m[2 * j], m[2 * j + 1]]);
}

function channelRoles() {
  const roles = Array.from({ length: NUM_CH }, () => []);
  readMapping().forEach(([pos, neg], j) => {
    if (pos >= 0) roles[pos].push(`${names[j]} +`);
    if (neg >= 0) roles[neg].push(`${names[j]} −`);
  });
  return roles.map((r) => r.join(', ') || 'unused');
}

function channelOptions(selected) {
  let html = `<option value="-1"${selected < 0 ? ' selected' : ''}>—</option>`;
  for (let ch = 0; ch < NUM_CH; ch++) {
    html += `<option value="${ch}"${ch === selected ? ' selected' : ''}>CH${ch + 1} · ${MOVES[ch].short}</option>`;
  }
  return html;
}

const mappingBody = $('mapping').querySelector('tbody');
readMapping().forEach(([pos, neg], j) => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${names[j]}</td>
    <td><select data-dir="pos">${channelOptions(pos)}</select></td>
    <td><select data-dir="neg">${channelOptions(neg)}</select></td>`;
  tr.addEventListener('change', () => {
    const [p, n] = [...tr.querySelectorAll('select')].map((s) => Number(s.value));
    app.set_mapping(j, p, n);
    refreshRoles();
  });
  mappingBody.append(tr);
});

function bindRange(id, setter, digits) {
  const input = $(id);
  const out = $(`${id}-val`);
  const apply = () => {
    setter(Number(input.value));
    out.textContent = Number(input.value).toFixed(digits);
  };
  input.addEventListener('input', apply);
  apply();
}
// ---- decoder ---------------------------------------------------------------

let decoderMode = 'pattern';

function refreshDecoderHint() {
  $('decoder-hint').textContent = decoderMode === 'direct'
    ? 'Each joint follows its own pair of channels, so crosstalk between electrodes also moves other joints.'
    : app.decoder_trained()
      ? 'Splits the activity across all 8 channels into your calibrated movements, so crosstalk doesn\'t move other joints and two movements at once drive both joints.'
      : 'Trained by Full calibration. Until then each joint follows its own pair of channels.';
}

function setDecoder(mode) {
  decoderMode = mode;
  app.set_pattern_decoder(mode === 'pattern');
  for (const b of $('decoder').querySelectorAll('button')) b.classList.toggle('active', b.dataset.mode === mode);
  refreshDecoderHint();
}

$('decoder').addEventListener('click', (e) => {
  const mode = e.target.dataset?.mode;
  if (mode) setDecoder(mode);
});

/** "Elbow +" etc.: where a channel's movement is mapped. */
function jointFor(ch) {
  const j = readMapping().findIndex(([pos, neg]) => pos === ch || neg === ch);
  if (j < 0) return 'not mapped';
  return `${names[j]} ${readMapping()[j][0] === ch ? '+' : '−'}`;
}

function drawIntent() {
  const el = $('intent');
  el.hidden = !(source === 'emg' && decoderMode === 'pattern' && app.decoder_trained());
  if (el.hidden) return;
  // [relaxed, (channel, intensity) × up to two simultaneous movements]
  const v = app.intent();
  const moves = [];
  for (let i = 1; i + 1 < v.length; i += 2) moves.push([v[i], v[i + 1]]);
  const pct = (w) => `${Math.round(Math.min(w, 1) * 100)}%`;
  el.classList.toggle('active', moves.length > 0);
  el.querySelector('.what').textContent = moves.length
    ? moves.map(([c]) => MOVES[c].cue).join(' + ')
    : v[0] === 1 ? 'Relaxed' : 'Below threshold';
  el.querySelector('.where').textContent = moves.length
    ? moves.map(([c, w]) => `${jointFor(c)} ${pct(w)}`).join(' · ')
    : v[0] === 1 ? 'no movement' : 'a movement is starting but not past the threshold';
  el.querySelector('.conf div').style.width = moves.length ? pct(Math.max(...moves.map((m) => m[1]))) : '0%';
}

bindRange('threshold', (v) => app.set_threshold(v), 2);
bindRange('gain', (v) => app.set_gain(v), 2);
$('mains').onchange = (e) => app.set_mains(Number(e.target.value));

// ---- channels panel --------------------------------------------------------

const channels = Array.from({ length: NUM_CH }, (_, ch) => {
  const row = document.createElement('div');
  row.className = 'chan';
  row.title = `Electrode site: ${MOVES[ch].muscle}`;
  row.innerHTML = `
    <div class="label">CH${ch + 1}<small></small></div>
    <div class="bar"><div class="fill"></div><div class="thr"></div></div>
    <div class="uv mono"></div>
    <button class="small" title="Recalibrate the ${MOVES[ch].muscle} pulse level">Pulse</button>`;
  row.querySelector('button').onclick = () => startCalibration(() => app.calibrate_max(ch));
  $('channels').append(row);
  return {
    row,
    role: row.querySelector('small'),
    fill: row.querySelector('.fill'),
    thr: row.querySelector('.thr'),
    uv: row.querySelector('.uv'),
  };
});

function refreshRoles() {
  channelRoles().forEach((r, ch) => { channels[ch].role.textContent = r; });
}
refreshRoles();

// ---- OpenBCI connection / simulator ----------------------------------------

let serial = null;
let sim = null;

const setBci = (msg) => { $('bci-status').textContent = msg; };

// A single-row cable shares one REF electrode, so subtracting the channel
// average cancels the hum and noise it puts on every channel.
const applyWiring = () => app.set_car($('wiring').value === 'common');
$('wiring').onchange = applyWiring;
const streaming = () => Boolean(sim || (serial && serial.binary));

function requireStream() {
  if (streaming()) return true;
  setCal('Connect the Cyton (or start Simulate) first.');
  return false;
}

function updateButtons() {
  $('connect').textContent = serial ? 'Disconnect' : 'Connect';
  $('simulate').textContent = sim ? 'Stop simulation' : 'Simulate';
  $('connect').disabled = Boolean(sim);
  $('simulate').disabled = Boolean(serial);
  $('wiring').disabled = Boolean(serial); // channel config is sent on connect
}

$('connect').onclick = async () => {
  if (serial) {
    await serial.disconnect();
    return;
  }
  if (!CytonSerial.supported) {
    setBci('Web Serial is unavailable. Use Chrome or Edge on http://localhost.');
    return;
  }
  serial = new CytonSerial({
    wiring: $('wiring').value,
    onData: (bytes) => app.push_serial(bytes),
    onStatus: setBci,
    onClose: () => {
      serial = null;
      stopCalibration();
      updateButtons();
      setBci('Disconnected');
    },
  });
  updateButtons();
  applyWiring();
  app.reset_stream();
  try {
    await serial.connect();
    setSource('emg');
  } catch (err) {
    const msg = err.name === 'NotFoundError' ? 'No port selected.' : err.message;
    await serial?.disconnect();
    serial = null;
    setBci(msg);
  }
  updateButtons();
};

$('simulate').onclick = () => {
  if (sim) {
    sim = null;
    stopCalibration();
    setBci('Not connected');
  } else {
    applyWiring();
    app.reset_stream();
    sim = new CytonSimulator();
    setSource('emg');
    setBci('Simulating a Cyton. Hold the joint keys to contract the mapped channels.');
  }
  updateButtons();
};

const guide = $('guide');
$('guide-open').onclick = () => { guide.hidden = false; };
guide.onclick = () => { guide.hidden = true; };
addEventListener('keydown', (e) => {
  if (e.key === 'Escape') guide.hidden = true;
});

function pumpSimulator(now) {
  const levels = new Float32Array(NUM_CH);
  const level = shift ? 1 : SIM_LEVEL;
  readMapping().forEach(([pos, neg], j) => {
    if (pos >= 0 && held.has(KEYS[j][0])) levels[pos] = level;
    if (neg >= 0 && held.has(KEYS[j][1])) levels[neg] = level;
  });
  // play along with a calibration: pulse the cued channel while it says PULSE
  const s = app.calibration_status();
  if (s.length && s[2] >= 0 && s[3] === 1) levels[s[2]] = SIM_PULSE;
  levels.forEach((v, ch) => sim.setLevel(ch, v));
  const bytes = sim.pump(now);
  if (bytes.length) app.push_serial(bytes);
}

// ---- guided calibration ------------------------------------------------------
//
// The schedule runs inside the WASM pipeline and is clocked by incoming samples,
// so the cue on screen and the recorded window can't drift apart. This code
// only draws what calibration_status() reports.

const setCal = (msg) => { $('cal-status').textContent = msg; };
const PHASE = { PREPARE: 0, RECORD: 1, RELAX: 2 };
const coach = {
  el: $('coach'),
  card: $('coach').querySelector('.card'),
  plan: [], // -1 = rest, otherwise the channel to pulse
  active: false,
  lastKey: '',
  lastPackets: 0,
  lastDataAt: 0,
};

function calSummary() {
  return app.is_calibrated()
    ? 'Calibrated to your pulse levels.'
    : 'Not fully calibrated: run Full calibration once the electrodes are on.';
}

let audio = null;
function beep(freq, ms = 110, volume = 0.06) {
  try {
    audio ??= new AudioContext();
    const osc = audio.createOscillator();
    const gain = audio.createGain();
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(volume, audio.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + ms / 1000);
    osc.connect(gain).connect(audio.destination);
    osc.start();
    osc.stop(audio.currentTime + ms / 1000);
  } catch { /* sound is optional */ }
}

function showCoachParts(running) {
  for (const id of ['coach-count', 'coach-live', 'coach-timeline', 'coach-cancel', 'coach-check']) $(id).hidden = !running;
  if (!running) $('coach-skip').hidden = true;
  for (const id of ['coach-results', 'coach-close']) $(id).hidden = running;
  $('coach-bar').parentElement.hidden = !running;
}

function startCalibration(begin) {
  if (!requireStream()) return;
  begin();
  coach.plan = Array.from(app.calibration_plan());
  if (!coach.plan.length) return;
  coach.active = true;
  coach.skipped = new Set();
  coach.wasClean = null;
  coach.lastKey = '';
  coach.lastDataAt = performance.now();
  $('coach-timeline').innerHTML = coach.plan.map(() => '<div></div>').join('');
  showCoachParts(true);
  coach.el.hidden = false;
  beep(660); // inside the click, which also unlocks audio for later cues
}

function stopCalibration() {
  app.cancel_calibration();
  coach.active = false;
  coach.el.hidden = true;
}

// Signal check reported with each calibration status (see App::calibration_status).
const CHECK = { NONE: -1, CLEAN: 0, NO_CONTACT: 1, MOVING: 2, NO_PULSE: 3, NOT_RELAXED: 4 };
/** Seconds without progress before the guide suggests skipping. */
const STUCK_HINT_S = 6;

const site = (c) => `CH${c + 1} (${MOVES[c].muscle})`;

/** Mapped channels whose electrode is offline right now. */
function offlineUsed() {
  const contact = app.contact();
  return [...new Set(readMapping().flat())].filter((c) => c >= 0 && !contact[c]).sort((a, b) => a - b);
}

/** [css class, message] for the live signal check. */
function checkMessage(check, checkCh, phase, rest, proxy, ch) {
  if (proxy && phase === PHASE.RECORD && (check === CHECK.CLEAN || check === CHECK.NO_PULSE)) {
    return check === CHECK.CLEAN
      ? ['clean', `Electrode ${site(ch)} offline: pulse read from the other channels`]
      : ['wait', `Electrode ${site(ch)} offline: pulse not visible on the other channels yet, tense a little harder`];
  }
  switch (check) {
    case CHECK.CLEAN: {
      if (phase === PHASE.RELAX) return ['clean', 'Relaxed'];
      if (!rest) return ['clean', 'Clean pulse: recording'];
      const off = offlineUsed();
      return ['clean', off.length
        ? `Clean: holding still (${off.map((c) => `CH${c + 1}`).join(', ')} offline, will be estimated from the others)`
        : 'Clean: holding still'];
    }
    case CHECK.NO_CONTACT:
      return ['bad', `No contact on ${site(checkCh)}: press the electrode down or re-gel it`];
    case CHECK.MOVING:
      return ['bad', `Movement or noise on ${site(checkCh)}: stay still (paused)`];
    case CHECK.NO_PULSE:
      return ['wait', 'No pulse yet: tense a little harder (paused)'];
    case CHECK.NOT_RELAXED:
      return ['wait', `Waiting for ${site(checkCh)} to relax`];
    default:
      return ['', ''];
  }
}

function drawCoach(now) {
  if (!coach.active) return;
  const s = app.calibration_status();
  if (!s.length) {
    finishCalibration();
    return;
  }
  const [step, steps, ch, phase, phaseProgress, secondsLeft, , check, checkCh, meter, stuck, proxy] = s;
  const rest = ch < 0;
  const move = rest ? null : MOVES[ch];
  const next = coach.plan[step + 1];

  // the clock is the sample stream: if the board stops, say so instead of freezing silently
  const packets = app.stats()[0];
  if (packets !== coach.lastPackets) {
    coach.lastPackets = packets;
    coach.lastDataAt = now;
  }
  const stalled = now - coach.lastDataAt > 700;

  let cls, label, title, detail;
  if (phase === PHASE.PREPARE) {
    [cls, label] = ['prepare', 'GET READY'];
    title = rest ? 'Relax your whole arm' : `Next: ${move.cue}`;
    detail = rest ? 'Rest it on the table, hands off the keyboard.' : `${move.muscle} · CH${ch + 1}, ${move.how}`;
  } else if (phase === PHASE.RECORD) {
    [cls, label] = rest ? ['rest', 'STAY RELAXED'] : ['hold', 'PULSE: TENSE NOW'];
    title = rest ? 'Stay completely relaxed' : move.cue;
    detail = rest
      ? 'Measuring your resting noise floor. Only still, clean seconds count.'
      : 'Tense as if about to move, but keep the arm still. Only clean pulse seconds count.';
  } else {
    [cls, label] = ['relax', 'RELAX'];
    title = 'Let go';
    detail = next === undefined ? 'Last one done' : `Next: ${next < 0 ? 'relax' : MOVES[next].cue}`;
  }
  const stuckLong = phase !== PHASE.PREPARE && stuck >= STUCK_HINT_S;
  if (stuckLong) detail = 'Stuck here? Fix the electrode, or skip this step and redo it later with its Pulse button.';
  if (stalled) detail = 'Paused: no data from the board';

  coach.card.className = `card ${cls}`;
  $('coach-step').textContent = `Step ${step + 1} of ${steps}`;
  $('coach-phase').textContent = label;
  $('coach-title').textContent = title;
  $('coach-detail').textContent = detail;
  const count = Math.ceil(secondsLeft - 1e-3);
  $('coach-count').textContent = count;
  $('coach-bar').style.width = `${phaseProgress * 100}%`;

  // live signal check: it decides whether this phase's countdown runs
  const [checkCls, checkText] = checkMessage(check, checkCh, phase, rest, proxy === 1, ch);
  $('coach-check').className = checkCls;
  $('coach-check').querySelector('.text').textContent = checkText;
  const skip = $('coach-skip');
  skip.hidden = phase === PHASE.PREPARE;
  skip.classList.toggle('urge', stuckLong);

  // meter: the checked level against its limit (the marker)
  const env = app.envelopes();
  const fill = $('coach-live').querySelector('.fill');
  if (check === CHECK.NONE) {
    const shown = rest ? env.indexOf(Math.max(...env)) : ch;
    const floor = Math.max(app.rest_levels()[shown], 0.5);
    $('coach-live-label').textContent = rest
      ? `Loudest channel: CH${shown + 1} · ${env[shown].toFixed(1)} µV`
      : `${move.muscle}: ${env[ch].toFixed(1)} µV · ${(env[ch] / floor).toFixed(1)}× rest`;
    fill.style.width = `${Math.min(env[shown] / (floor * 12), 1) * 100}%`;
    $('coach-live').querySelector('.floor').style.left = `${100 / 12}%`;
  } else {
    // marker at the middle = the limit; pulses must reach it, rest/relax must stay under it
    $('coach-live-label').textContent = phase === PHASE.RECORD && !rest
      ? proxy === 1
        ? `Strongest other channel at ${meter.toFixed(1)}× the level needed`
        : `${move.muscle}: ${env[ch].toFixed(1)} µV · ${meter.toFixed(1)}× the level needed`
      : `Busiest channel at ${meter.toFixed(1)}× its ${rest ? 'stillness' : 'relaxed'} limit`;
    fill.style.width = `${Math.min(meter / 2, 1) * 100}%`;
    $('coach-live').querySelector('.floor').style.left = '50%';
  }
  fill.classList.toggle('clean', check === CHECK.CLEAN);

  [...$('coach-timeline').children].forEach((d, i) => {
    d.className = i < step ? 'done' : i === step ? 'now' : '';
  });

  // sound on the same clock: ticks for the last 3 s of GET READY, a tone at each switch,
  // and a low buzz when a recording loses its clean signal
  const key = `${step}:${phase}:${count}`;
  if (key !== coach.lastKey && !stalled) {
    const [pStep, pPhase] = coach.lastKey.split(':').map(Number);
    if (pStep !== step || pPhase !== phase) {
      if (phase === PHASE.RECORD) beep(rest ? 520 : 880, 180);
      else if (phase === PHASE.RELAX) beep(440, 140);
    } else if (phase === PHASE.PREPARE && count <= 3) {
      beep(600, 50, 0.04);
    }
    coach.lastKey = key;
  }
  const clean = check === CHECK.CLEAN;
  if (phase === PHASE.RECORD && coach.wasClean === true && !clean && !stalled) beep(200, 160, 0.05);
  coach.wasClean = phase === PHASE.RECORD ? clean : null;
}

function finishCalibration() {
  coach.active = false;
  beep(990, 220);
  const rest = app.rest_levels();
  const max = app.max_levels();
  const contact = app.contact();
  const accepted = app.max_calibrated();
  const viaOthers = app.calibrated_via_others();
  const pulsed = coach.plan.filter((c) => c >= 0);

  // decoder recognition per class, tested on held-out halves of the recordings
  const decoded = new Map();
  const report = app.decoder_report();
  for (let i = 0; i < report.length; i += 3) decoded.set(report[i], { acc: report[i + 1], with: report[i + 2] });
  const nameOf = (code) => (code === -1 ? 'relaxed' : MOVES[code]?.short ?? '?');
  const decodeCell = (code) => {
    const d = decoded.get(code);
    if (!d) return '<span class="bad">not trained</span>';
    const pct = `${Math.round(d.acc * 100)}%`;
    if (d.acc >= 0.9) return `<span class="good">${pct}</span>`;
    const mix = d.with > -2 ? ` · mixed up with ${nameOf(d.with)}` : '';
    return `<span class="${d.acc >= 0.7 ? 'ok' : 'bad'}">${pct}${mix}</span>`;
  };

  const rows = pulsed.map((ch) => {
    const ratio = max[ch] / Math.max(rest[ch], 0.1);
    // a rejected pulse leaves the channel deliberately insensitive and out of the decoder
    const [cls, verdict] = coach.skipped.has(ch) ? ['bad', accepted[ch] ? 'skipped (kept old)' : 'skipped']
      : viaOthers[ch] ? ['ok', 'offline · via others']
      : !contact[ch] ? ['bad', 'no contact']
      : !accepted[ch] ? ['bad', 'no pulse seen']
        : ratio >= 4 ? ['good', 'clear']
          : ratio >= 2 ? ['ok', 'usable']
            : ['bad', 'weak'];
    const shown = accepted[ch] ? `${ratio.toFixed(1)}×` : '—';
    return `<div class="res"><span>${MOVES[ch].short} · ${MOVES[ch].muscle}</span>`
      + `<span class="mono">${shown}</span><span class="${cls}">${verdict}</span>${decodeCell(ch)}</div>`;
  });
  if (decoded.size) {
    rows.unshift(`<div class="res"><span>Relaxed</span><span></span><span></span>${decodeCell(-1)}</div>`);
  }
  $('coach-results').innerHTML = pulsed.length
    ? '<div class="res head"><span>Movement</span><span>pulse</span><span>signal</span><span>decoded</span></div>' + rows.join('')
    : '';
  const avgRest = rest.reduce((a, b) => a + b, 0) / rest.length;

  coach.card.className = 'card done';
  $('coach-step').textContent = 'Calibration complete';
  $('coach-phase').textContent = 'DONE';
  $('coach-title').textContent = pulsed.length ? 'Pulse levels recorded' : 'Rest level recorded';
  $('coach-detail').textContent = pulsed.length
    ? '"Decoded" is how often the decoder recognised each movement on data it wasn\'t trained on. '
      + 'Movements that get mixed up need a more distinct pulse or electrode spot; redo one with its channel\'s Pulse button.'
    : `Average resting floor ${avgRest.toFixed(1)} µV.`;
  showCoachParts(false);
  setCal(calSummary());
  refreshDecoderHint();
}

$('cal-all').onclick = () => startCalibration(() => app.calibrate_full());
$('cal-rest').onclick = () => startCalibration(() => app.calibrate_rest());
$('coach-cancel').onclick = () => {
  stopCalibration();
  setCal('Calibration cancelled.');
};
$('coach-close').onclick = () => { coach.el.hidden = true; };
$('coach-skip').onclick = () => {
  const s = app.calibration_status();
  if (!s.length) return;
  // a pulse step skipped during its relax phase was already recorded
  if (s[3] !== PHASE.RELAX) coach.skipped.add(s[2]);
  app.skip_calibration_step();
};
addEventListener('keydown', (e) => {
  if (e.key !== 'Escape' || coach.el.hidden) return;
  if (coach.active) $('coach-cancel').click();
  else coach.el.hidden = true;
});

// ---- main loop -------------------------------------------------------------

let last = performance.now();
let lastUi = 0;
let lastStats = { t: last, packets: 0 };

function updateUi(now) {
  const angles = app.joint_angles();
  const inputs = app.joint_inputs();
  joints.forEach((j, i) => {
    j.angle.textContent = `${fmt(angles[i])}°`;
    const v = inputs[i];
    j.meter.style.left = `${50 + Math.min(v, 0) * 50}%`;
    j.meter.style.width = `${Math.abs(v) * 50}%`;
  });

  const [x, y, z] = app.end_effector();
  $('ee').innerHTML = `<span>End effector (m)</span><span>${fmt(x, 3)} ${fmt(y, 3)} ${fmt(z, 3)}</span>`
    + `<span>Claw</span><span>${app.holding() ? 'holding · set down to drop' : 'open · touch box to grab'}</span>`;

  const env = app.envelopes();
  const act = app.activations();
  const thr = Number($('threshold').value);
  const contact = app.contact();
  const estimated = app.estimated();
  const cal = app.calibration_status();
  const recording = cal.length > 0 && cal[3] === PHASE.RECORD;
  channels.forEach((c, ch) => {
    c.fill.style.width = `${act[ch] * 100}%`;
    c.fill.classList.toggle('on', act[ch] > thr);
    c.thr.style.left = `${thr * 100}%`;
    c.uv.textContent = estimated[ch] ? `~${env[ch].toFixed(1)}` : contact[ch] ? `${env[ch].toFixed(1)}µV` : 'off';
    c.row.classList.toggle('estimated', Boolean(estimated[ch]));
    c.row.classList.toggle('nocontact', !contact[ch] && !estimated[ch]);
    c.row.title = contact[ch]
      ? `Electrode site: ${MOVES[ch].muscle}`
      : estimated[ch]
        ? `${MOVES[ch].muscle}: electrode offline, estimated from the other channels. Press it down or re-gel it.`
        : `${MOVES[ch].muscle}: no contact (input railed). Press the electrode down or re-gel it.`;
    c.row.classList.toggle('calibrating', recording && (cal[2] < 0 || cal[2] === ch));
  });

  const [packets, dropped] = app.stats();
  if (now - lastStats.t >= 1000) {
    $('pps').textContent = Math.round(((packets - lastStats.packets) * 1000) / (now - lastStats.t));
    lastStats = { t: now, packets };
  }
  $('drops').textContent = dropped;

  const pill = $('mode-pill');
  const uncal = source === 'emg' && !app.is_calibrated();
  pill.textContent = source === 'emg' ? (uncal ? 'EMG · uncalibrated' : 'EMG') : 'Manual';
  pill.className = `pill ${source === 'emg' ? (uncal ? 'warn' : 'emg') : ''}`;
}

function frame(now) {
  const dt = (now - last) / 1000;
  last = now;

  if (source === 'manual') {
    joints.forEach((j, i) => app.set_input(i, keyAxis(i) || Number(j.slider.value)));
  }
  if (sim) pumpSimulator(now);
  app.frame(dt);
  drawCoach(now);
  drawIntent();

  if (now - lastUi >= UI_INTERVAL_MS) {
    lastUi = now;
    updateUi(now);
  }
  requestAnimationFrame(frame);
}

setSource('manual');
setDecoder('pattern');
updateButtons();
requestAnimationFrame(frame);
