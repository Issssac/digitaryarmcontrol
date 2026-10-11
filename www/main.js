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

// Finger grid (see finger-grid.svg): CH1–4 in the upper row, about a quarter of
// the way from the elbow crease to the wrist, CH5–8 in the middle row, about
// halfway. The order within a row doesn't matter: calibration learns each
// finger's pattern from whatever site ends up on which channel.
const GRID = [...Array(4).fill('Upper row'), ...Array(4).fill('Middle row')];
const FINGERS = [
  { name: 'Thumb', the: 'your thumb', how: 'bend its tip toward the palm' },
  { name: 'Index', the: 'your index finger', how: 'curl it toward the palm' },
  { name: 'Middle', the: 'your middle finger', how: 'curl it toward the palm' },
  { name: 'Ring + little', the: 'your ring and little fingers', how: 'curl both together' },
];
const NUM_FINGERS = FINGERS.length;
// Rest postures recorded in calibration, so a raised arm isn't read as finger tension.
const POSTURES = [
  { name: 'limp', title: 'Relax your whole hand', place: 'Forearm resting on the table, palm up, fingers loose, hands off the keyboard.' },
  { name: 'forearm raised', title: 'Raise your forearm, elbow on the table', place: 'Forearm up, elbow resting on the table, hand limp.' },
  { name: 'arm up', title: 'Hold your arm up, unsupported', place: 'Lift the whole arm off the table and hold it there, hand limp.' },
];
// Simulator only: the arm posture to fake when no calibration is cueing one (keys 1/2/3).
let simPosture = 0;
// finger_states() / goal codes
const BAND = { REST: 0, PENDING: 1, MINUS: 2, PLUS: 3, OVER: 4 };
// finger_status() codes
const FSTEP = { REST: 0, HARD: 1, MEDIUM: 2, FAINT: 3, TRAIN: 4 };
const FCHECK = { NONE: -1, CLEAN: 0, NO_CONTACT: 1, MOVING: 2, WEAK: 3, STRONG: 4, OTHER: 5, NOT_RELAXED: 6, FAINTER: 7 };
/** On-screen meters are averaged over about this long, so they don't jitter. */
const DISPLAY_TAU_S = 0.25;
const FMETER = { ACTIVITY: 0, OF_HARD: 1, POWER: 2, STILL: 3 };
/** Finger bars run from rest (0) past the ceiling (1) to this. */
const POWER_MAX = 1.25;

let scheme = 'muscle';

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
  $('source-hint').textContent = src !== 'emg'
    ? 'Keys Q/A · W/S · E/D · R/F, or drag the sliders (they spring back to 0).'
    : scheme === 'finger'
      ? 'Arm follows finger tension. With Simulate on, Q/W/E/R = firmer (+) and A/S/D/F = faint (−) on the finger driving that joint (Shift = a real movement); 1/2/3 = arm limp / forearm raised / arm up.'
      : 'Arm follows muscle activation. With Simulate on, Q/A · W/S · E/D · R/F fake contractions (Shift = full).';
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
  if (['1', '2', '3'].includes(k)) simPosture = Number(k) - 1; // simulated arm posture
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
$('drift').onchange = (e) => app.set_drift(e.target.checked);

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

function drawFingerIntent(el) {
  el.hidden = !(source === 'emg' && app.finger_trained());
  if (el.hidden) return;
  const states = app.finger_states();
  const joints = app.finger_joints();
  const powers = app.finger_powers();
  const over = states.includes(BAND.OVER);
  const active = FINGERS.map((_, f) => f).filter((f) => states[f] === BAND.MINUS || states[f] === BAND.PLUS);
  const plus = (f) => states[f] === BAND.PLUS;
  el.classList.toggle('active', active.length > 0 && !over);
  el.querySelector('.what').textContent = over
    ? 'Hand moved: holding'
    : active.length ? active.map((f) => `${FINGERS[f].name} ${plus(f) ? 'firm' : 'faint'}`).join(' + ') : 'Relaxed';
  const pose = postureText();
  el.querySelector('.where').textContent = (over
    ? 'a finger crossed its ceiling, so every joint holds'
    : active.length ? active.map((f) => `${names[joints[f]]} ${plus(f) ? '+' : '−'}`).join(' · ') : 'no finger past its rest band')
    + (pose ? ` · arm: ${pose}` : '');
  el.querySelector('.conf div').style.width = pctOf(Math.max(...powers), 1);
}

function drawIntent() {
  const el = $('intent');
  if (scheme === 'finger') {
    drawFingerIntent(el);
    return;
  }
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
  const roles = scheme === 'finger' ? GRID : channelRoles();
  roles.forEach((r, ch) => { channels[ch].role.textContent = r; });
}
refreshRoles();

/** Where a channel's electrode sits in the current scheme. */
const siteName = (ch) => (scheme === 'finger' ? `grid ${GRID[ch].toLowerCase()}` : MOVES[ch].muscle);

// ---- finger grid -------------------------------------------------------------

const pctOf = (v, max) => `${Math.max(0, Math.min(v / max, 1)) * 100}%`;

/**
 * Power bar: coloured zones [from, to, class] on a 0..max scale, an optional
 * target box [lo, hi], and the live marker. Zones are rebuilt only when they change.
 */
function drawTrack(el, max, zones, target, value, on = false) {
  const key = `${max}|${zones.map((z) => z.join(',')).join(';')}`;
  if (el.dataset.key !== key) {
    el.dataset.key = key;
    el.innerHTML = zones
      .map(([a, b, cls]) => `<div class="zone ${cls}" style="left:${pctOf(a, max)};width:calc(${pctOf(b, max)} - ${pctOf(a, max)})"></div>`)
      .join('') + '<div class="target"></div><div class="marker"></div>';
  }
  const box = el.querySelector('.target');
  box.style.display = target ? '' : 'none';
  if (target) {
    box.style.left = pctOf(target[0], max);
    box.style.width = `calc(${pctOf(Math.min(target[1], max), max)} - ${pctOf(target[0], max)})`;
  }
  const marker = el.querySelector('.marker');
  marker.style.left = pctOf(value, max);
  marker.classList.toggle('on', on);
}

function fingerBands(f) {
  const b = app.finger_bands();
  const o = f * 7;
  return {
    stasis: b[o], restExit: b[o + 1], level: b[o + 2], split: b[o + 3],
    plusEnter: b[o + 4], plusExit: b[o + 5], plusTarget: b[o + 6],
  };
}

/** rest (incl. the buffer) · faint − · firmer + · real movement (no ceiling with simple bands) */
const bandZones = (b) => (app.finger_simple_bands()
  ? [[0, b.restExit, 'rest'], [b.restExit, b.split, 'minus'], [b.split, POWER_MAX, 'plus']]
  : [[0, b.restExit, 'rest'], [b.restExit, b.split, 'minus'], [b.split, 1, 'plus'], [1, POWER_MAX, 'over']]);
/** finger_report() values per finger. */
const REPORT_LEN = 13;
const fingerCalibrated = (f) => app.finger_report()[f * REPORT_LEN] === 1;

const STATE_TEXT = ['rest', 'faint…', 'faint −', 'firm +', 'moved'];
const STATE_CLS = ['', 'minus', 'minus', 'plus', 'over'];

const fingerRows = FINGERS.map((fg, f) => {
  const row = document.createElement('div');
  row.className = 'finger uncal';
  row.innerHTML = `
    <div class="name">${fg.name}<small></small></div>
    <select aria-label="Joint driven by ${fg.name}">${names.map((n, j) => `<option value="${j}">${n}</option>`).join('')}</select>
    <div class="state mono"></div>
    <button class="small" title="Recalibrate and retrain just this finger">Redo</button>
    <div class="ftrack"></div>`;
  const select = row.querySelector('select');
  select.value = String(app.finger_joints()[f]);
  select.onchange = () => app.set_finger_joint(f, Number(select.value));
  row.querySelector('button').onclick = () => startCalibration(() => app.finger_calibrate_one(f, true), 'finger');
  $('fingers').append(row);
  return { row, state: row.querySelector('.state'), track: row.querySelector('.ftrack'), note: row.querySelector('small') };
});

/** "forearm raised 85%": the posture the decoder thinks the arm is in, or '' before calibration. */
function postureText() {
  const m = app.finger_posture();
  if (!m.length) return '';
  const top = m.indexOf(Math.max(...m));
  return `${POSTURES[top].name} ${Math.round(m[top] * 100)}%`;
}

function drawFingerRows() {
  const pose = postureText();
  $('fposture').innerHTML = pose ? `<span>Arm posture</span><span>${pose}</span>` : '';
  const powers = app.finger_powers();
  const states = app.finger_states();
  const rep = app.finger_report();
  const simple = app.finger_simple_bands();
  fingerRows.forEach((r, f) => {
    const cal = fingerCalibrated(f);
    const b = fingerBands(f);
    const smoothMs = rep[f * REPORT_LEN + 11];
    r.row.classList.toggle('uncal', !cal);
    drawTrack(r.track, POWER_MAX, cal ? bandZones(b) : [[0, POWER_MAX, 'rest']], null, powers[f], states[f] >= BAND.MINUS);
    r.state.textContent = cal ? STATE_TEXT[states[f]] : '—';
    r.state.className = `state mono ${cal ? STATE_CLS[states[f]] : ''}`;
    r.note.textContent = !cal ? 'not calibrated'
      : (simple ? '− ¼–½ · + over ½' : `faint ${Math.round(b.level * 100)}% · firm ${Math.round(b.plusTarget * 100)}%`)
        + (smoothMs > 65 ? ` · ${Math.round(smoothMs)} ms` : '');
  });
}

function refreshFingerStatus() {
  const trained = app.finger_trained();
  $('ftrain').disabled = !trained;
  $('fcal-status').textContent = trained
    ? 'Calibrated. Train only practises the levels again; they get fainter as you get consistent. Redo recalibrates one finger.'
    : 'Not calibrated. Calibrate + train takes about 5 minutes: rest, then each finger from a hard flex down to faint tension, then practice moving the arm.';
}

function setScheme(s) {
  if (coach.active) stopCalibration();
  scheme = s;
  app.set_finger_mode(s === 'finger');
  document.body.classList.toggle('fingers', s === 'finger');
  for (const b of $('scheme').querySelectorAll('button')) b.classList.toggle('active', b.dataset.scheme === s);
  $('scheme-hint').textContent = s === 'finger'
    ? 'Eight electrodes in a 2×4 grid on the underside of the forearm. Each finger drives one joint: faint tension = −, firmer = +, and a real finger movement is ignored.'
    : 'One electrode per muscle; each joint follows an antagonist pair of muscles.';
  $('guide-img').src = s === 'finger' ? 'finger-grid.svg' : 'electrodes.svg';
  try { localStorage.setItem('scheme', s); } catch { /* remembering is optional */ }
  refreshRoles();
  refreshFingerStatus();
  setSource(source);
}

$('scheme').addEventListener('click', (e) => {
  const s = e.target.dataset?.scheme;
  if (s && s !== scheme) setScheme(s);
});
bindRange('fspeed', (v) => app.set_finger_speed(v), 2);
// the decoder is the source of truth: never let a checkbox restored by the browser disagree with it
$('fsimple').checked = app.finger_simple_bands();
$('fsimple').onchange = (e) => app.set_finger_simple_bands(e.target.checked);
$('fcal').onclick = () => startCalibration(() => app.finger_calibrate(true), 'finger');
$('ftrain').onclick = () => startCalibration(() => app.finger_train(), 'finger');

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
  $('fcal-status').textContent = 'Connect the Cyton (or start Simulate) first.';
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
    sim = new CytonSimulator({ differential: $('wiring').value === 'bipolar' });
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

// Simulated finger tension (1 = a hard flex). Like a person, the simulated
// user steers it by watching a reading: the coach's bar, or the decoded power.
const simFinger = new Float32Array(NUM_FINGERS);

function steerSim(f, aim, reading, guess, gain = 0.012) {
  const d = simFinger[f] || guess;
  simFinger[f] = Math.max(0.001, Math.min(1.2, d + gain * (aim - reading)));
}

function simulateFingers() {
  const steered = new Set();
  const s = app.finger_status();
  // the cued rest posture during calibration (limp for everything else), otherwise keys 1/2/3
  sim.setPosture(s.length ? Math.max(s[18], 0) : simPosture);
  if (s.length) {
    // play along with the coach
    const [step, f, phase, , , , , , kind, meter, lo, hi] = s;
    if (phase === PHASE.RECORD && step !== FSTEP.REST) {
      if (kind === FMETER.ACTIVITY) {
        simFinger[f] = 1;
      } else {
        const aim = Number.isFinite(s[17]) ? s[17] : lo + 0.45 * (hi - lo);
        steerSim(f, aim, meter, kind === FMETER.OF_HARD ? aim : aim * 0.45);
      }
      steered.add(f);
    }
  } else if (app.finger_trained()) {
    // keys: + key = firmer, − key = faint, on the finger that drives that joint
    const powers = app.finger_powers();
    const joints = app.finger_joints();
    for (let f = 0; f < NUM_FINGERS; f++) {
      const [up, down] = KEYS[joints[f]].map((k) => held.has(k));
      if (!up && !down) continue;
      const b = fingerBands(f);
      if (shift) simFinger[f] = 1;
      // a deliberate press: starts near the right tension and settles quickly
      else steerSim(f, up ? b.plusTarget : b.level, powers[f], up ? 0.3 : 0.12, 0.02);
      steered.add(f);
    }
  }
  simFinger.forEach((_, f) => {
    if (!steered.has(f)) simFinger[f] = 0;
    sim.setFingerLevel(f, simFinger[f]);
  });
}

function pumpSimulator(now) {
  const levels = new Float32Array(NUM_CH);
  if (scheme === 'finger') {
    simulateFingers();
  } else {
    const level = shift ? 1 : SIM_LEVEL;
    readMapping().forEach(([pos, neg], j) => {
      if (pos >= 0 && held.has(KEYS[j][0])) levels[pos] = level;
      if (neg >= 0 && held.has(KEYS[j][1])) levels[neg] = level;
    });
    // play along with a calibration: pulse the cued channel while it says PULSE
    const s = app.calibration_status();
    if (s.length && s[2] >= 0 && s[3] === 1) levels[s[2]] = SIM_PULSE;
    simFinger.fill(0);
    simFinger.forEach((v, f) => sim.setFingerLevel(f, v));
  }
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
  kind: 'muscle', // or 'finger'
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
  for (const id of ['coach-count', 'coach-timeline', 'coach-cancel', 'coach-check']) $(id).hidden = !running;
  $('coach-live').hidden = !running || coach.kind === 'finger';
  $('fbar').hidden = !running || coach.kind !== 'finger';
  if (!running) $('coach-skip').hidden = true;
  for (const id of ['coach-results', 'coach-close']) $(id).hidden = running;
  $('coach-bar').parentElement.hidden = !running;
}

/** kind: 'muscle' (muscle-pair calibration) or 'finger' (finger calibration / training). */
function startCalibration(begin, kind = 'muscle') {
  if (!requireStream()) return;
  begin();
  let segments;
  if (kind === 'finger') {
    const s = app.finger_status();
    if (!s.length) return;
    segments = s[13];
  } else {
    coach.plan = Array.from(app.calibration_plan());
    if (!coach.plan.length) return;
    segments = coach.plan.length;
  }
  coach.kind = kind;
  coach.active = true;
  coach.skipped = new Set();
  coach.wasClean = null;
  coach.lastKey = '';
  coach.lastProgress = 0;
  coach.lastDataAt = performance.now();
  $('coach-timeline').innerHTML = '<div></div>'.repeat(segments);
  showCoachParts(true);
  coach.el.hidden = false;
  beep(660); // inside the click, which also unlocks audio for later cues
}

function stopCalibration() {
  app.cancel_calibration();
  app.finger_cancel();
  coach.active = false;
  coach.el.hidden = true;
  $('trainer').hidden = true;
}

/** The clock is the sample stream: if the board stops, say so instead of freezing silently. */
function isStalled(now) {
  const packets = app.stats()[0];
  if (packets !== coach.lastPackets) {
    coach.lastPackets = packets;
    coach.lastDataAt = now;
  }
  return now - coach.lastDataAt > 700;
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
  if (coach.kind === 'finger') {
    drawFingerCoach(now);
    return;
  }
  const s = app.calibration_status();
  if (!s.length) {
    finishCalibration();
    return;
  }
  const [step, steps, ch, phase, phaseProgress, secondsLeft, , check, checkCh, meter, stuck, proxy] = s;
  const rest = ch < 0;
  const move = rest ? null : MOVES[ch];
  const next = coach.plan[step + 1];
  const stalled = isStalled(now);

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

// ---- finger coach ----------------------------------------------------------------

/** [css class, message] for the finger coach's live check. */
function fingerCheckMessage(check, arg, step, phase, goal, f) {
  switch (check) {
    case FCHECK.CLEAN:
      if (phase === PHASE.RELAX) return ['clean', 'Relaxed'];
      if (step === FSTEP.REST) return ['clean', 'Clean: holding still'];
      return ['clean', step === FSTEP.TRAIN ? 'On target: hold it' : 'Clean: recording'];
    case FCHECK.NO_CONTACT:
      return ['bad', `Too few electrodes with contact: CH${arg + 1} (${GRID[arg].toLowerCase()}) is off`];
    case FCHECK.MOVING:
      return ['bad', `Movement on CH${arg + 1}: stay still (paused)`];
    case FCHECK.WEAK:
      if (step === FSTEP.HARD) return ['wait', 'Not enough yet: flex harder'];
      if (step === FSTEP.MEDIUM) return ['wait', 'A bit more'];
      if (step === FSTEP.FAINT) return ['wait', 'Below the noise: just a touch more, to the left edge of the box'];
      return ['wait', goal === BAND.MINUS ? 'Not there yet: a faint tension' : 'More: firmer, still without moving'];
    case FCHECK.STRONG:
      if (step === FSTEP.MEDIUM) return ['bad', 'Nearly as hard as the hard flex: ease off'];
      if (step === FSTEP.TRAIN && goal === BAND.MINUS && app.finger_states()[f] !== BAND.OVER) {
        return ['bad', 'Too firm: that\'s the + band, ease off'];
      }
      return ['bad', 'Over the ceiling: that\'s a real movement, ease off'];
    case FCHECK.FAINTER:
      return ['wait', 'A little fainter: under the right edge of the box'];
    case FCHECK.OTHER:
      return ['wait', `${FINGERS[arg].name} is stronger: keep the other fingers loose`];
    case FCHECK.NOT_RELAXED:
      return ['wait', step === FSTEP.TRAIN ? `Let ${FINGERS[f].the} go` : `Waiting for CH${arg + 1} to relax`];
    default:
      return ['', ''];
  }
}

const LEGEND = {
  rest: ['#2a2e36', 'rest'],
  minus: ['#1d3f63', 'faint −'],
  plus: ['#1e4d36', 'firmer +'],
  over: ['#5a2323', 'real movement'],
  ok: ['#1e4d36', 'counts'],
  weak: ['#2a2e36', 'noise'],
};

function drawFingerBar(kind, meter, lo, hi, step, f, clean) {
  let max;
  let zones;
  let label;
  if (kind === FMETER.ACTIVITY) {
    max = 3;
    zones = [[0, 1, 'weak'], [1, 3, 'ok']];
    label = `Flex strength: ${meter.toFixed(1)}× what's needed`;
  } else if (kind === FMETER.OF_HARD) {
    max = 1;
    zones = [[0, 0.15, 'weak'], [0.15, 0.85, 'ok'], [0.85, 1, 'over']];
    label = `${Math.round(meter * 100)}% of your hard flex · aim for the white box`;
  } else if (kind === FMETER.POWER) {
    const b = fingerBands(f);
    if (step === FSTEP.FAINT) {
      // zoomed in on the bottom of the range, so a box just above the noise has room
      max = Math.min(POWER_MAX, Math.max(0.25, hi * 2.5));
      zones = app.finger_simple_bands()
        ? [[0, lo, 'weak'], [lo, max, 'ok']]
        : [[0, lo, 'weak'], [lo, Math.min(1, max), 'ok'], [1, POWER_MAX, 'over']];
    } else {
      max = POWER_MAX;
      zones = bandZones(b);
    }
    label = `${FINGERS[f].name}: ${Math.round(meter * 100)}% of the ceiling (where the finger really moves)`
      + (max < 1 ? ` · bar shows 0–${Math.round(max * 100)}%` : '');
  } else {
    max = 2;
    zones = [[0, 1, 'ok'], [1, 2, 'over']];
    label = `Busiest channel at ${meter.toFixed(1)}× its resting limit`;
  }
  const target = Number.isFinite(lo) ? [lo, hi] : null;
  drawTrack($('fbar').querySelector('.ftrack'), max, zones, target, meter, clean);
  $('fbar-label').textContent = label;
  const legend = [...new Set(zones.map((z) => z[2]))].map((c) => `<span><i style="background:${LEGEND[c][0]}"></i>${LEGEND[c][1]}</span>`);
  if (target) legend.push('<span><i style="border:2px solid #fff;box-sizing:border-box"></i>aim here</span>');
  $('fbar-legend').innerHTML = legend.join('');
}

function drawFingerCoach(now) {
  const s = app.finger_status();
  if (!s.length) {
    finishFingerCalibration();
    return;
  }
  const [step, f, phase, progress, secondsLeft, check, arg, stuck, kind, meter, lo, hi, segment, , round, rounds, goal, aim, posture] = s;
  const fg = FINGERS[f];
  const joint = f >= 0 ? names[app.finger_joints()[f]] : '';
  const stalled = isStalled(now);
  const count = Math.ceil(secondsLeft - 1e-3);

  // averaged for display; restarts with every step so it doesn't slide in from the last one
  const phaseKey = `${step}:${f}:${round}:${phase}`;
  const dt = Math.min((now - (coach.lastNow ?? now)) / 1000, 0.1);
  coach.lastNow = now;
  if (coach.smoothKey !== phaseKey || !Number.isFinite(coach.smooth)) {
    coach.smoothKey = phaseKey;
    coach.smooth = meter;
  }
  coach.smooth += (meter - coach.smooth) * (1 - Math.exp(-dt / DISPLAY_TAU_S));
  const shown = coach.smooth;

  // phase 2 runs in a small meter over the arm; phase 1 in the full card
  const training = step === FSTEP.TRAIN;
  coach.el.hidden = training;
  $('trainer').hidden = !training;
  if (training) {
    drawTrainer({ f, joint, phase, progress, check, round, rounds, goal, lo, hi, aim, shown, stalled });
  } else {
    drawFingerCard({ step, f, phase, progress, count, check, arg, stuck, kind, shown, lo, hi, segment, round, rounds, goal, stalled, posture });
  }

  // sounds on the sample clock: a tone at each switch, ticks before a recording,
  // and for training a chirp on a hit and a low tone on a miss
  const key = `${step}:${f}:${round}:${phase}:${count}`;
  if (key !== coach.lastKey && !stalled) {
    const [pStep, pF, pRound, pPhase] = coach.lastKey.split(':').map(Number);
    if (pStep !== step || pF !== f || pRound !== round || pPhase !== phase) {
      if (pStep === FSTEP.TRAIN && pPhase === PHASE.RECORD && phase === PHASE.RELAX) {
        if (coach.lastProgress >= 0.9) beep(1250, 130);
        else beep(260, 220, 0.05);
      } else if (phase === PHASE.RECORD) beep(step === FSTEP.REST ? 520 : 880, 180);
      else if (phase === PHASE.RELAX) beep(440, 140);
    } else if (phase === PHASE.PREPARE && count <= 3 && step !== FSTEP.TRAIN) {
      beep(600, 50, 0.04);
    }
    coach.lastKey = key;
  }
  coach.lastProgress = progress;
  const clean = check === FCHECK.CLEAN;
  if (phase === PHASE.RECORD && step !== FSTEP.TRAIN && coach.wasClean === true && !clean && !stalled) beep(200, 160, 0.05);
  coach.wasClean = phase === PHASE.RECORD ? clean : null;
}

/**
 * Phase 2: a small meter over the arm. The centre line is the target; the fill
 * grows red to the right when over it and green to the left when under, and
 * each end of the meter is the edge of the band the trial asks for.
 */
function drawTrainer({ f, joint, phase, progress, check, round, rounds, goal, lo, hi, aim, shown, stalled }) {
  const fg = FINGERS[f];
  const minus = goal === BAND.MINUS;
  const el = $('trainer');
  const active = phase === PHASE.RECORD;
  el.classList.toggle('idle', !active);
  $('trainer-what').textContent = `${fg.name} ${minus ? 'faint' : 'firmer'} → ${joint} ${minus ? '−' : '+'}`;
  $('trainer-trial').textContent = `trial ${round + 1}/${rounds}`;
  $('trainer-hint').textContent = stalled ? 'Paused: no data from the board'
    : phase === PHASE.PREPARE ? (minus ? 'Next: the faintest tension you can feel' : 'Next: a little firmer, finger still')
      : phase === PHASE.RELAX ? (check === FCHECK.CLEAN ? 'Relaxed' : `Let ${fg.the} go`)
        : app.finger_states()[f] === BAND.OVER ? 'Over the ceiling: that was a real movement'
          : 'Bring it to the centre line and hold';
  const fill = el.querySelector('.t-fill');
  if (Number.isFinite(aim)) {
    const over = shown >= aim;
    const frac = Math.min(1, over ? (shown - aim) / Math.max(hi - aim, 1e-3) : (aim - shown) / Math.max(aim - lo, 1e-3));
    fill.className = `t-fill ${over ? 'over' : 'under'}`;
    fill.style.left = over ? '50%' : `${50 - frac * 50}%`;
    fill.style.width = `${frac * 50}%`;
  } else {
    fill.style.width = '0';
  }
  el.querySelector('.t-hold div').style.width = `${active ? progress * 100 : 0}%`;
}

/** Phase 1 (and rest): the full coach card with the power bar. */
function drawFingerCard({ step, f, phase, progress, count, check, arg, stuck, kind, shown, lo, hi, segment, round, rounds, goal, stalled, posture }) {
  const fg = FINGERS[f];
  const pose = POSTURES[posture];
  let cls;
  let label;
  let title;
  let detail;
  if (step === FSTEP.REST) {
    const prep = phase === PHASE.PREPARE;
    [cls, label] = prep ? ['prepare', posture ? 'CHANGE POSITION' : 'GET READY'] : ['rest', 'STAY RELAXED'];
    title = prep ? pose.title : posture ? 'Hold it there, hand relaxed' : 'Stay completely relaxed';
    detail = prep
      ? pose.place
      : posture
        ? 'Measuring how this position changes the resting signal, so it isn\'t read as finger tension.'
        : 'Measuring the resting noise on every electrode. Only still seconds count.';
  } else if (phase === PHASE.PREPARE) {
    [cls, label] = ['prepare', 'GET READY'];
    title = step === FSTEP.HARD ? `Next: ${fg.name}, a hard flex`
      : step === FSTEP.MEDIUM ? `Next: ${fg.name}, about half as hard`
        : `Next: ${fg.name}, ${round ? 'even fainter' : 'faint'}${rounds > 1 ? ` (round ${round + 1}, up to ${rounds})` : ''}`;
    detail = step === FSTEP.FAINT
      ? 'Tension only this time: don\'t let the finger move.'
      : `${step === FSTEP.HARD ? 'Forearm back on the table, palm up. ' : ''}This one is a real movement: ${fg.how}.`;
  } else if (phase === PHASE.RECORD) {
    if (step === FSTEP.HARD) {
      [cls, label] = ['real', 'FLEX HARD'];
      title = `Flex ${fg.the} hard`;
      detail = `${fg.how[0].toUpperCase()}${fg.how.slice(1)} and hold. Keep the other fingers loose.`;
    } else if (step === FSTEP.MEDIUM) {
      [cls, label] = ['real', 'FLEX HALF'];
      title = 'About half as hard';
      detail = 'Still a real movement, just gentler.';
    } else {
      [cls, label] = ['faint', rounds > 1 ? `TENSE ONLY · ROUND ${round + 1}` : 'TENSE ONLY'];
      title = round ? 'Even fainter' : 'As faint as you can';
      detail = 'Don\'t move the finger. Barely tense it: keep the line in the box, just past its left edge, '
        + 'which sits just above your resting noise.';
    }
  } else {
    [cls, label] = ['relax', 'RELAX'];
    title = 'Let go';
    detail = 'Waiting for every channel to settle.';
  }
  const stuckLong = phase !== PHASE.PREPARE && stuck >= STUCK_HINT_S && step !== FSTEP.REST;
  if (stuckLong) {
    detail = step === FSTEP.FAINT ? 'Can\'t hold it there? Skip stops here and keeps the faintest level so far.'
      : 'Stuck? Fix the electrode, or skip this finger.';
  }
  if (stalled) detail = 'Paused: no data from the board';

  coach.card.className = `card ${cls}`;
  // phase 1 goes round by round: every finger strong, then half, then faint
  const roundName = step === FSTEP.HARD ? 'strong flex' : step === FSTEP.MEDIUM ? 'half strength' : 'faint';
  const roundNo = step === FSTEP.HARD ? 1 : step === FSTEP.MEDIUM ? 2 : 3 + round;
  $('coach-step').textContent = step === FSTEP.REST
    ? `Finger calibration · rest ${posture + 1} of ${POSTURES.length}: ${pose.name}`
    : `Round ${roundNo} of ${2 + rounds || 3} · ${roundName} · ${fg.name}`;
  $('coach-phase').textContent = label;
  $('coach-title').textContent = title;
  $('coach-detail').textContent = detail;
  $('coach-count').textContent = count;
  $('coach-bar').style.width = `${progress * 100}%`;

  const [checkCls, checkText] = fingerCheckMessage(check, arg, step, phase, goal, f);
  $('coach-check').className = checkCls;
  $('coach-check').querySelector('.text').textContent = checkText;
  const skip = $('coach-skip');
  // the limp rest is the reference for everything else, so it can't be skipped
  skip.hidden = step === FSTEP.REST && posture === 0;
  skip.textContent = step === FSTEP.REST ? 'Skip this position'
    : step === FSTEP.FAINT ? 'Stop here (keep faintest)' : 'Skip this finger';
  skip.classList.toggle('urge', stuckLong);

  drawFingerBar(kind, shown, lo, hi, step, f, check === FCHECK.CLEAN);
  [...$('coach-timeline').children].forEach((d, i) => {
    d.className = i < segment ? 'done' : i === segment ? 'now' : '';
  });
}

function finishFingerCalibration() {
  coach.active = false;
  $('trainer').hidden = true;
  coach.el.hidden = false;
  beep(990, 220);
  const rep = app.finger_report();
  const joints = app.finger_joints();
  const rows = FINGERS.map((fg, f) => {
    const [cal, faint, , isolation, hits, trials, , , , heldFaint, heldFirm, smoothMs, mixed] =
      rep.slice(f * REPORT_LEN, (f + 1) * REPORT_LEN);
    const name = `<span>${fg.name} → ${names[joints[f]]}</span>`;
    if (!cal) return `<div class="res fres">${name}<span class="bad">skipped</span><span></span><span></span><span></span><span></span></div>`;
    const grade = (v, good, ok) => (v >= good ? 'good' : v >= ok ? 'ok' : 'bad');
    const training = trials ? `<span class="${grade(hits / trials, 0.75, 0.5)}">${hits}/${trials} hits</span>` : '<span>—</span>';
    const pct = (v) => (v > 0 ? Math.round(v * 100) : '–');
    return `<div class="res fres">${name}<span class="mono">${Math.round(faint * 100)}%</span>`
      + `<span class="${mixed ? 'bad' : grade(isolation, 0.9, 0.7)}" title="${mixed ? 'The faint round looked more like another finger, so it was not used' : ''}">`
      + `${Math.round(isolation * 100)}%${mixed ? ' · mixed' : ''}</span>${training}`
      + `<span class="mono">${pct(heldFaint)} / ${pct(heldFirm)}%</span><span class="mono">${Math.round(smoothMs)} ms</span></div>`;
  });
  $('coach-results').innerHTML = '<div class="res fres head"><span>Finger → joint</span><span>faintest</span>'
    + '<span>isolated</span><span>training</span><span>you held</span><span>smoothing</span></div>' + rows.join('');
  coach.card.className = 'card done';
  $('coach-step').textContent = 'Finger calibration complete';
  $('coach-phase').textContent = 'DONE';
  $('coach-title').textContent = 'Levels set';
  $('coach-detail').textContent = 'Faintest: how far below the ceiling (where the finger really moves) each finger still read clearly. '
    + 'Isolated: how often that faint tension read as the right finger ("mixed": the faint round looked like another finger, so it wasn\'t used). '
    + 'You held: where your faint and firm attempts '
    + 'really averaged (raw); the decoder now maps them onto the middle of the − and + bands. '
    + 'Smoothing: how much it filters that finger to cancel your wobble. Train only again refines both. '
    + 'Redo a finger from its row in the panel.';
  showCoachParts(false);
  refreshFingerStatus();
}

$('cal-all').onclick = () => startCalibration(() => app.calibrate_full());
$('cal-rest').onclick = () => startCalibration(() => app.calibrate_rest());
$('coach-cancel').onclick = () => {
  stopCalibration();
  setCal('Calibration cancelled.');
};
$('coach-close').onclick = () => { coach.el.hidden = true; };
$('coach-skip').onclick = () => {
  if (coach.kind === 'finger') {
    app.finger_skip();
    return;
  }
  const s = app.calibration_status();
  if (!s.length) return;
  // a pulse step skipped during its relax phase was already recorded
  if (s[3] !== PHASE.RELAX) coach.skipped.add(s[2]);
  app.skip_calibration_step();
};
$('trainer-skip').onclick = () => app.finger_skip();
$('trainer-stop').onclick = () => $('coach-cancel').click();
addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
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
      ? `Electrode site: ${siteName(ch)}`
      : estimated[ch]
        ? `${siteName(ch)}: electrode offline, estimated from the other channels. Press it down or re-gel it.`
        : `${siteName(ch)}: no contact (input railed). Press the electrode down or re-gel it.`;
    c.row.classList.toggle('calibrating', recording && (cal[2] < 0 || cal[2] === ch));
  });

  const [packets, dropped] = app.stats();
  if (now - lastStats.t >= 1000) {
    $('pps').textContent = Math.round(((packets - lastStats.packets) * 1000) / (now - lastStats.t));
    lastStats = { t: now, packets };
  }
  $('drops').textContent = dropped;

  if (scheme === 'finger') drawFingerRows();

  const pill = $('mode-pill');
  const uncal = source === 'emg' && !(scheme === 'finger' ? app.finger_trained() : app.is_calibrated());
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
let savedScheme = 'muscle';
try { savedScheme = localStorage.getItem('scheme') === 'finger' ? 'finger' : 'muscle'; } catch { /* default */ }
setScheme(savedScheme);
updateButtons();
requestAnimationFrame(frame);
