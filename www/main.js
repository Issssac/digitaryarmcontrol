import init, { App } from './pkg/arm_sim.js';
import { CytonSerial } from './serial.js';
import { CytonSimulator, DIGIT_PATTERNS } from './simulator.js';
import { RoArm, SERVOS_DOWN } from './roarm.js';
import { CalibrationRecorder } from './recorder.js';

const NUM_CH = 8;
// [+ key, − key] per joint: base, shoulder, elbow, gripper (+ opens)
const KEYS = [['a', 'd'], ['w', 's'], ['f', 'r'], ['q', 'e']];
// Muscle-pair gestures, in the decoder's order (see DEFAULT_MAPPING in
// pipeline.rs for the axis each drives), and the muscle that mostly carries
// each. Calibration learns each one's pattern across all channels, so it doesn't
// matter which electrode is on which muscle. Tense as if starting the movement,
// with the arm held still.
const GESTURES = [
  { muscle: 'Pronator teres', short: 'Rotate in', cue: 'Rotate your wrist in', how: 'palm turning down, as if pouring out a cup', anim: 'gesture-rotate-in.svg' },
  { muscle: 'Supinator', short: 'Rotate out', cue: 'Rotate your wrist out', how: 'palm turning up, as if turning a key', anim: 'gesture-rotate-out.svg' },
  { muscle: 'Triceps', short: 'Tricep extend', cue: 'Straighten your arm', how: 'as if pressing your forearm down into the table', anim: 'gesture-tricep.svg' },
  { muscle: 'Biceps', short: 'Bicep curl', cue: 'Bicep curl', how: 'palm up, as if lifting a cup', anim: 'gesture-bicep.svg' },
  { muscle: 'Wrist extensors', short: 'Wrist back', cue: 'Pull your wrist back', how: 'knuckles up, as if revving a motorbike', anim: 'gesture-wrist-back.svg' },
  { muscle: 'Wrist flexors', short: 'Wrist curl', cue: 'Curl your wrist in', how: 'palm side toward the forearm', anim: 'gesture-wrist-curl.svg' },
  { muscle: 'Finger flexors', short: 'Grip', cue: 'Grip', how: 'close your fingers as if squeezing a ball', anim: 'gesture-grip.svg' },
];
// What the muscle pairs drive (reach.rs): the reach axis takes the shoulder's
// slot and moves the gripper tip along where it points.
const AXES = ['Base', 'Reach', 'Elbow', 'Gripper'];
/** The grip gesture, and the simulated finger-extensor channel it co-contracts. */
const GRIP = 6;
const SIM_FINGER_EXTENSORS = 7;
const SIM_GRIP_BRACE = 0.5;
const SIM_LEVEL = 0.7;
// Simulated tension during a calibration pulse (lands below a real max, like a light intention).
const SIM_PULSE = 0.5;
const UI_INTERVAL_MS = 66;

// Digit array (see digit-array.svg): eight bipolar pairs. CH1–3 on the finger
// flexors and CH4–6 on the finger extensors, each from the thumb side to the
// little-finger side, CH7 on the biceps, CH8 on the triceps. Movements in the
// decoder's order (DIGIT_AXES in gesture.rs for the axis each drives).
const DIGIT_SITES = ['Flexors · thumb side', 'Flexors · middle', 'Flexors · little side',
  'Extensors · thumb side', 'Extensors · middle', 'Extensors · little side', 'Biceps', 'Triceps'];
const DIGITS = [
  { short: 'Thumb flex', cue: 'Flex your thumb', how: 'bend the thumb tip in toward your palm' },
  { short: 'Thumb extend', cue: 'Lift your thumb out', how: 'straighten the thumb back, away from the palm' },
  { short: 'Index flex', cue: 'Curl your index finger', how: 'bend it toward the palm, the other fingers loose' },
  { short: 'Index extend', cue: 'Lift your index finger', how: 'straighten it up and back, the other fingers loose' },
  { short: 'Ring + little flex', cue: 'Curl your ring and little fingers', how: 'bend both toward the palm together' },
  { short: 'Ring + little extend', cue: 'Lift your ring and little fingers', how: 'straighten both up and back together' },
  { short: 'Bicep', cue: 'Bicep curl', how: 'palm up, as if lifting a cup', anim: 'gesture-bicep.svg' },
  { short: 'Tricep', cue: 'Straighten your arm', how: 'as if pressing your forearm down into the table', anim: 'gesture-tricep.svg' },
];
// Rest postures recorded in calibration, so a raised arm isn't read as finger tension.
const POSTURES = [
  { name: 'limp', title: 'Relax your whole hand', place: 'Forearm resting on the table, palm up, fingers loose, hands off the keyboard.', img: 'posture-limp.svg' },
  { name: 'forearm raised', title: 'Raise your forearm, elbow on the table', place: 'Forearm up, elbow resting on the table, hand limp.', img: 'posture-forearm.svg' },
  { name: 'arm up', title: 'Hold your arm up, unsupported', place: 'Lift the whole arm off the table and hold it there, hand limp.', img: 'posture-up.svg' },
];
// Simulator only: the arm posture to fake when no calibration is cueing one (keys 1/2/3).
let simPosture = 0;
// gesture_status() codes
const FSTEP = { REST: 0, HARD: 1, MEDIUM: 2, FAINT: 3, TRAIN: 4 };
const FCHECK = { NONE: -1, CLEAN: 0, NO_CONTACT: 1, MOVING: 2, WEAK: 3, STRONG: 4, OTHER: 5, NOT_RELAXED: 6, FAINTER: 7 };
/** On-screen meters are averaged over about this long, so they don't jitter. */
const DISPLAY_TAU_S = 0.25;
const FMETER = { ACTIVITY: 0, OF_HARD: 1, POWER: 2, STILL: 3 };
/** Power bars run from rest (0) past the ceiling (1) to this. */
const POWER_MAX = 1.25;

let scheme = 'muscle'; // or 'digit'
const isDigit = () => scheme === 'digit';
/** The current scheme's movements, in the decoder's order. */
const moves = () => (isDigit() ? DIGITS : GESTURES);

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

// every calibration run is recorded to recordings/<id>/ (see recorder.js, serve.py)
const recorder = new CalibrationRecorder({
  packets: () => app.stats()[0],
  onError: (msg) => { $('saved-status').textContent = msg; },
});
/** All stream bytes go through here: to the parser, and to an active recording. */
function feedStream(bytes) {
  recorder.capture(bytes);
  app.push_serial(bytes);
}

const names = app.joint_names();
/** The gripper has an open/close button; keys and EMG drive the other joints. */
const GRIPPER = 3;
const DRIVEN = names.slice(0, GRIPPER);
const jointName = (j) => (j >= 0 ? names[j] : 'off');

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
// EMG drives the arm while the Cyton (or the simulator) is streaming; otherwise
// it's manual, from the joint buttons and keys.

let source = 'manual';

function setSource(src) {
  source = src;
  app.set_use_emg(src === 'emg');
  $('emg-state').textContent = src === 'emg' ? (sim ? 'Simulated EMG Engaged' : 'EMG Engaged') : 'Manual Mode Engaged';
  $('emg-state').classList.toggle('emg', src === 'emg');
  if (src === 'emg') requireStream(); // clears a "connect first" warning
  $('source-hint').textContent = src !== 'emg'
    ? ''
    : isDigit()
      ? 'Arm follows the digit array: thumb flex/extend closes/opens the gripper, index flex/extend tips it down/up, ring + little flex/extend turn the base, bicep/tricep reach in/out. With Simulate on: A/D base, W/S reach out/in, F/R tip, Q/E gripper (Shift = full).'
      : 'Arm follows your gestures: wrist back/curl aim the elbow, tricep/bicep reach out/in along the gripper, and it closes while you grip. With Simulate on: A/D base, W/S reach out/in, F/R elbow, E grip (Shift = full).';
}

// ---- keyboard --------------------------------------------------------------

const held = new Set();
let shift = false;
const ALL_KEYS = KEYS.flat();

// Space is the emergency stop, wherever focus is. Captured and cancelled on both
// key events, so it can't also press a focused button (Engage among them).
addEventListener('keydown', (e) => {
  if (e.key !== ' ' || e.target.tagName === 'INPUT') return; // a space in the run name is just a space
  e.preventDefault();
  e.stopImmediatePropagation();
  if (e.repeat) return;
  emergencyStop();
  coachReady();
}, { capture: true });
addEventListener('keyup', (e) => {
  if (e.key === ' ' && e.target.tagName !== 'INPUT') e.preventDefault();
}, { capture: true });

addEventListener('keydown', (e) => {
  shift = e.shiftKey;
  if (e.ctrlKey || e.metaKey || e.altKey || ['SELECT', 'INPUT'].includes(e.target.tagName)) return;
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

/** The gripper moves toward open or closed while Q / E (or its buttons) are held, and stops on release. */
let gripperAxis = 0;
function driveGripper() {
  const axis = keyAxis(GRIPPER);
  if (axis === gripperAxis) return;
  gripperAxis = axis;
  if (axis) app.set_gripper(axis > 0);
  else app.hold_gripper();
}

// ---- joint buttons -----------------------------------------------------------
// A column per joint: its angle, the real arm's while engaged, then + over −.
// Holding a button is holding its key, so buttons and keys share one path
// (and with Simulate on, the arm buttons fake contractions like the keys do).

const joints = names.map((name, j) => {
  const col = document.createElement('div');
  col.className = 'jcol';
  const [plus, minus] = KEYS[j];
  const verbs = j === GRIPPER ? ['open', 'close'] : ['+', '−'];
  col.innerHTML = `
    <div class="jname">${name}</div>
    <div class="jangle mono"></div>
    <div class="jreal mono"></div>
    <button class="jbtn" tabindex="-1" data-key="${plus}" title="Hold to ${verbs[0] === '+' ? 'move +' : verbs[0]} (${plus.toUpperCase()})">+<small>${plus.toUpperCase()}</small></button>
    <button class="jbtn" tabindex="-1" data-key="${minus}" title="Hold to ${verbs[1] === '−' ? 'move −' : verbs[1]} (${minus.toUpperCase()})">−<small>${minus.toUpperCase()}</small></button>`;
  $('jgrid').append(col);
  return {
    angle: col.querySelector('.jangle'),
    real: col.querySelector('.jreal'),
    buttons: [...col.querySelectorAll('.jbtn')],
  };
});

for (const b of $('jgrid').querySelectorAll('.jbtn')) {
  const release = () => held.delete(b.dataset.key);
  b.addEventListener('pointerdown', (e) => {
    e.preventDefault(); // a long press mustn't select text or open a touch menu
    b.setPointerCapture(e.pointerId);
    held.add(b.dataset.key);
  });
  b.addEventListener('pointerup', release);
  b.addEventListener('pointercancel', release);
  b.addEventListener('lostpointercapture', release);
}

$('reset').onclick = () => app.reset();
$('clear-trail').onclick = () => app.clear_trail();
let roarm = null;

// Never while the real arm follows: it would creep home on its own.
let drift = true;
const applyDrift = () => app.set_drift(drift && !roarm?.following);
function setDrift(on) {
  drift = on;
  $('drift').classList.toggle('active', on);
  $('drift').textContent = on ? 'Drift ON' : 'Drift OFF';
  $('drift').setAttribute('aria-pressed', String(on));
  applyDrift();
}
$('drift').onclick = () => {
  setDrift(!drift);
  saveSoon();
};

// ---- mapping ---------------------------------------------------------------

/** [axis, direction] per movement of each scheme, fixed in gesture.rs. */
const SCHEME_AXES = Object.fromEntries([['muscle', false], ['digit', true]].map(([name, digit]) => {
  const v = app.scheme_axes(digit);
  return [name, Array.from({ length: v.length / 2 }, (_, g) => [v[2 * g], v[2 * g + 1]])];
}));
const axesOf = () => SCHEME_AXES[scheme];

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
/** "Elbow +", "Reach out", "Gripper close" etc.: what a movement drives. */
function jointFor(g) {
  const [j, d] = axesOf()[g];
  const plus = d > 0;
  const dir = j === GRIPPER ? (plus ? 'open' : 'close') : j === 1 ? (plus ? 'out' : 'in')
    : j === 2 && isDigit() ? (plus ? '+ (tip down)' : '− (tip up)') : plus ? '+' : '−';
  return `${AXES[j]} ${dir}`;
}

function drawIntent() {
  const el = $('intent');
  el.hidden = !(source === 'emg' && app.gesture_trained());
  if (el.hidden) return;
  // every gesture driving its axis, and how far toward full speed (its level)
  const active = app.gesture_active();
  const powers = app.gesture_powers();
  const bands = app.gesture_bands();
  const speed = (g) => Math.min(powers[g] / bands[3 * g + 2], 1);
  const on = moves().map((_, g) => g).filter((g) => active[g]);
  const pct = (w) => `${Math.round(w * 100)}%`;
  el.classList.toggle('active', on.length > 0);
  el.querySelector('.what').textContent = on.length ? on.map((g) => moves()[g].short).join(' + ') : 'Relaxed';
  const pose = postureText(app.gesture_posture());
  el.querySelector('.where').textContent = (on.length ? on.map((g) => `${jointFor(g)} ${pct(speed(g))}`).join(' · ') : `no ${SET.noun} past its rest band`)
    + (pose ? ` · arm: ${pose}` : '');
  el.querySelector('.conf div').style.width = on.length ? pct(Math.max(...on.map(speed))) : '0%';
}

bindRange('gain', (v) => app.set_gain(v), 2);

// ---- channels panel --------------------------------------------------------
// A 4×2 grid (see style.css for the order per scheme). Each cell: the channel
// and the gesture it responds to most (learned by calibration, since any
// electrode may be on any muscle), the joint that gesture drives and the
// channel's level, then the meter.

const channels = Array.from({ length: NUM_CH }, (_, ch) => {
  const row = document.createElement('div');
  row.className = 'chan';
  row.innerHTML = `
    <div class="chan-top"><span class="name"><b>CH${ch + 1}</b><span class="gesture"></span></span></div>
    <div class="chan-mid"><span class="role"></span><span class="uv mono"></span></div>
    <div class="bar"><div class="fill"></div></div>`;
  $('channels').append(row);
  return {
    row,
    gesture: row.querySelector('.gesture'),
    role: row.querySelector('.role'),
    fill: row.querySelector('.fill'),
    uv: row.querySelector('.uv'),
  };
});

/**
 * Per channel: its strongest calibrated movement and that movement's joint (for
 * the digit array, its site comes first, since each pair has a fixed muscle).
 */
function refreshRoles() {
  const learned = app.channel_gestures();
  channels.forEach((c, ch) => {
    const g = learned[ch];
    const move = g >= 0 ? moves()[g]?.short ?? '' : '';
    c.gesture.textContent = isDigit() ? DIGIT_SITES[ch] : move;
    c.role.textContent = g >= 0 ? (isDigit() ? `${move} → ${jointFor(g)}` : jointFor(g))
      : app.gesture_calibrated() ? 'helper' : 'uncalibrated';
  });
}
refreshRoles();

/** Where a channel's electrode sits in the current scheme. */
const siteName = (ch) => (isDigit() ? `CH${ch + 1} (${DIGIT_SITES[ch].toLowerCase()})` : `CH${ch + 1}`);

// ---- power bars ----------------------------------------------------------------

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

/** "forearm raised 85%": the posture the decoder thinks the arm is in, or '' before calibration. */
function postureText(m) {
  if (!m.length) return '';
  const top = m.indexOf(Math.max(...m));
  return `${POSTURES[top].name} ${Math.round(m[top] * 100)}%`;
}

/** Calibrated / Uncalibrated, for the current scheme. */
function drawCalPills() {
  const done = app.gesture_calibrated();
  const pill = $('cal-pill');
  pill.textContent = done ? 'Calibrated' : 'Uncalibrated';
  pill.className = `pill ${done ? 'good' : 'warn'}`;
}

function setScheme(s) {
  if (coach.active) stopCalibration('scheme changed');
  scheme = s;
  // the digit array is bipolar: each pair cancels the common noise, so no channel averaging
  app.set_digit_mode(isDigit());
  document.body.classList.toggle('digit', isDigit());
  $('scheme').value = s;
  $('guide-img').src = isDigit() ? 'digit-array.svg' : 'electrodes.svg';
  try { localStorage.setItem('scheme', s); } catch { /* remembering is optional */ }
  if (sim) startSimulator(); // the simulated person wears the new electrodes
  if (serial && serial.wiring !== wiringFor()) {
    setBci(`Reconnect the Cyton: this scheme needs ${wiringFor()} wiring, and it was set up for ${serial.wiring}.`);
  }
  refreshRoles();
  setSource(source);
}

$('scheme').onchange = (e) => setScheme(e.target.value);

// ---- OpenBCI connection / simulator ----------------------------------------

let serial = null;
let sim = null;

const setBci = (msg) => { $('bci-status').textContent = msg; };

// 60 Hz mains. Muscle pairs use a single-row cable sharing one REF electrode
// (common wiring, with the channel average subtracted to cancel the hum it puts
// on every channel); the digit array uses a pair per channel (bipolar, SRB2 off).
app.set_mains(60);
const wiringFor = () => (isDigit() ? 'bipolar' : 'common');

const streaming = () => Boolean(sim || (serial && serial.binary));

/** Calibration needs a stream; say so under the calibration buttons until there is one. */
function requireStream() {
  const ok = streaming();
  const msg = ok ? '' : 'Connect the Cyton (or start Simulate) first.';
  setCal(msg);
  return ok;
}

function updateButtons() {
  $('connect').textContent = serial ? 'DISCONNECT EMG' : 'CONNECT EMG';
  $('simulate').textContent = sim ? 'Stop simulation' : 'Simulate';
  $('connect').disabled = Boolean(sim);
  $('simulate').disabled = Boolean(serial);
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
    wiring: wiringFor(),
    onData: feedStream,
    onStatus: setBci,
    onClose: () => {
      serial = null;
      stopCalibration('stream closed');
      setSource('manual');
      updateButtons();
      setBci('Disconnected');
    },
  });
  updateButtons();
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

/** A simulated Cyton wired for the current scheme (bipolar pairs see little common noise and crosstalk). */
function startSimulator() {
  app.reset_stream();
  sim = isDigit() ? new CytonSimulator({ differential: true, crosstalk: false, patterns: DIGIT_PATTERNS }) : new CytonSimulator();
}

$('simulate').onclick = () => {
  if (sim) {
    sim = null;
    stopCalibration('simulation stopped');
    setSource('manual');
    setBci('');
  } else {
    startSimulator();
    setSource('emg');
    setBci('Simulating a Cyton. Hold the joint keys or buttons to contract the mapped channels.');
  }
  updateButtons();
};

// The electrode guide: on its own from the Electrode guide buttons, or as the
// first page of a full calibration, which starts once the electrodes are confirmed.
const guide = $('guide');
let afterGuide = null;

function showGuide(then = null) {
  afterGuide = then;
  guide.classList.toggle('gate', Boolean(then));
  $('guide-gate').hidden = !then;
  $('guide-caption').textContent = then
    ? 'Put the electrodes on as shown, then continue to calibration.'
    : 'Click anywhere or press Esc to close';
  guide.hidden = false;
}

function closeGuide() {
  guide.hidden = true;
  afterGuide = null;
}

for (const b of document.querySelectorAll('.guide-open')) b.onclick = () => showGuide();
guide.onclick = () => { if (!afterGuide) closeGuide(); };
$('guide-cancel').onclick = closeGuide;
$('guide-ready').onclick = () => {
  const then = afterGuide;
  closeGuide();
  then();
};
addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeGuide();
});

// ---- the calibration set ---------------------------------------------------------
//
// Both schemes are calibrated the same way inside the WASM (gesture.rs): rest in
// three arm postures, then every movement strong → half → faint, then training.
// The schedule is clocked by incoming samples, so the cue on screen and the
// recorded window can't drift apart. The calls act on the current scheme.

const SET = {
  get label() { return isDigit() ? 'Digit array calibration' : 'Gesture calibration'; },
  get noun() { return isDigit() ? 'movement' : 'gesture'; },
  get count() { return moves().length; },
  status: () => app.gesture_status(),
  skip: () => app.gesture_skip(),
  cancel: () => app.gesture_cancel(),
  ready: () => app.gesture_ready(),
  anim: (g) => moves()[g]?.anim ?? null,
  trained: () => app.gesture_trained(),
  name: (g) => moves()[g].short,
  get restRead() { return isDigit() ? 'a finger movement' : 'a gesture'; },
  prepare: (step, g) => (step === FSTEP.FAINT
    ? 'Tension only this time: as if about to do it, with the arm and fingers still.'
    : `${step === FSTEP.HARD ? 'Arm back down, forearm on the table. ' : ''}${moves()[g].cue}: ${moves()[g].how}. A real contraction this time.`),
  hard: (g) => [`${moves()[g].cue}: hard`, `${cap(moves()[g].how)}, and hold. Keep the rest of the arm relaxed.`],
  other: (h) => `That reads as ${moves()[h].short}: keep the rest of the arm relaxed`,
  letGo: () => 'Let go',
  noContact: (c) => `Too few electrodes with contact: ${siteName(c)} is off`,
  ceiling: 'your half-strength contraction',
};
const cap = (s) => s[0].toUpperCase() + s.slice(1);

// ---- simulator ------------------------------------------------------------------
// Like a person, the simulated user steers each movement by watching a reading:
// the coach's bar, or the decoded power. Muscle pairs: gesture g's muscle sits
// under channel g (any placement works: calibration learns it), with grip also
// bracing the finger extensors under CH8. Digit array: each movement's share on
// every pair (DIGIT_PATTERNS in simulator.js).

const simDrive = new Float32Array(NUM_CH);

function steerSim(i, aim, reading, guess, gain = 0.012) {
  const d = simDrive[i] || guess;
  simDrive[i] = Math.max(0.001, Math.min(1.2, d + gain * (aim - reading)));
}

/** Simulated tension for each movement: playing along with the coach, or the keys once trained. */
function simulatePatterns() {
  const steered = new Set();
  const s = SET.status();
  // the cued rest posture during calibration (limp for everything else), otherwise keys 1/2/3
  sim.setPosture(s.length ? Math.max(s[18], 0) : simPosture);
  if (s.length) {
    const [step, i, phase, , , , , , meterKind, meter, lo, hi] = s;
    if (phase === PHASE.RECORD && step !== FSTEP.REST) {
      if (meterKind === FMETER.ACTIVITY) {
        simDrive[i] = 1;
      } else {
        const aim = Number.isFinite(s[17]) ? s[17] : lo + 0.45 * (hi - lo);
        steerSim(i, aim, meter, meterKind === FMETER.OF_HARD ? aim : aim * 0.45);
      }
      steered.add(i);
    }
  } else if (app.gesture_trained()) {
    // keys: the axis key in a movement's direction performs it, at a little over its level
    const powers = app.gesture_powers();
    const bands = app.gesture_bands();
    axesOf().forEach(([axis, dir], g) => {
      if (!held.has(KEYS[axis][dir > 0 ? 0 : 1])) return;
      if (shift) simDrive[g] = 1;
      else steerSim(g, bands[3 * g + 2] * 1.15, powers[g], 0.2, 0.02);
      steered.add(g);
    });
  }
  simDrive.forEach((_, i) => { if (!steered.has(i)) simDrive[i] = 0; });
}

function pumpSimulator(now) {
  simulatePatterns();
  if (isDigit()) {
    DIGITS.forEach((_, g) => sim.setPatternLevel(g, simDrive[g]));
  } else {
    const levels = new Float32Array(NUM_CH);
    GESTURES.forEach((_, g) => { levels[g] = simDrive[g]; });
    levels[SIM_FINGER_EXTENSORS] = Math.max(levels[SIM_FINGER_EXTENSORS], levels[GRIP] * SIM_GRIP_BRACE);
    levels.forEach((v, ch) => sim.setLevel(ch, v));
  }
  const bytes = sim.pump(now);
  if (bytes.length) feedStream(bytes);
}

// ---- calibration coach ----------------------------------------------------------

const setCal = (msg) => { $('cal-status').textContent = msg; };
const PHASE = { PREPARE: 0, RECORD: 1, RELAX: 2 };
/** Seconds without progress before the guide suggests skipping. */
const STUCK_HINT_S = 6;
const coach = {
  el: $('coach'),
  card: $('coach').querySelector('.card'),
  scheme: 'muscle', // the scheme being calibrated
  active: false,
  lastKey: '',
  lastPackets: 0,
  lastDataAt: 0,
};

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
  for (const id of ['coach-count', 'coach-timeline', 'coach-cancel', 'coach-check', 'fbar']) $(id).hidden = !running;
  if (!running) {
    $('coach-skip').hidden = true;
    $('coach-pose').hidden = true;
    $('coach-ready').hidden = true;
  }
  for (const id of ['coach-results', 'coach-close']) $(id).hidden = running;
  $('coach-bar').parentElement.hidden = !running;
}

/**
 * Starts a calibration run of the current scheme and records it.
 * mode: 'full', 'train only', 'redo …'; name: the run's name, from askRunName().
 */
function startCalibration(begin, mode, name) {
  if (!requireStream()) return;
  if (coach.active) stopCalibration('restarted');
  const before = toBase64(app.export_calibration());
  begin();
  const s = SET.status();
  if (!s.length) return;
  recorder.start(runMeta(name, mode, before));
  coach.lastLogKey = '';
  coach.lastLogAt = 0;
  coach.scheme = scheme;
  coach.active = true;
  coach.wasClean = null;
  coach.lastKey = '';
  coach.lastProgress = 0;
  coach.lastDataAt = performance.now();
  $('coach-timeline').innerHTML = '<div></div>'.repeat(s[13]);
  showCoachParts(true);
  coach.el.hidden = false;
  beep(660); // inside the click, which also unlocks audio for later cues
}

function stopCalibration(reason = 'cancelled') {
  if (coach.active) recordEnd(reason);
  app.gesture_cancel();
  coach.active = false;
  coach.el.hidden = true;
  $('trainer').hidden = true;
  $('trainer-anim').hidden = true;
}

/** Space or the Ready button: the user is in position, so the step's countdown starts. */
function coachReady() {
  if (!coach.active) return;
  const set = SET;
  const s = set.status();
  if (s.length && s[19] === 1) {
    recorder.event('ready');
    set.ready();
  }
}
$('coach-ready-btn').addEventListener('click', coachReady);

/** The clock is the sample stream: if the board stops, say so instead of freezing silently. */
function isStalled(now) {
  const packets = app.stats()[0];
  if (packets !== coach.lastPackets) {
    coach.lastPackets = packets;
    coach.lastDataAt = now;
  }
  return now - coach.lastDataAt > 700;
}

/** [css class, message] for the coach's live check. */
function checkMessage(set, check, arg, step, phase, i) {
  switch (check) {
    case FCHECK.CLEAN:
      if (phase === PHASE.RELAX) return ['clean', 'Relaxed'];
      if (step === FSTEP.REST) return ['clean', 'Clean: holding still'];
      return ['clean', step === FSTEP.TRAIN ? 'On target: hold it' : 'Clean: recording'];
    case FCHECK.NO_CONTACT:
      return ['bad', set.noContact(arg)];
    case FCHECK.MOVING:
      return ['bad', `Movement on CH${arg + 1}: stay still (paused)`];
    case FCHECK.WEAK:
      if (step === FSTEP.HARD) return ['wait', 'Not enough yet: harder'];
      if (step === FSTEP.MEDIUM) return ['wait', 'A bit more'];
      if (step === FSTEP.FAINT) return ['wait', 'Below the noise: just a touch more, to the left edge of the box'];
      return ['wait', 'More: a little firmer, arm still'];
    case FCHECK.STRONG:
      if (step === FSTEP.MEDIUM) return ['bad', 'Nearly as hard as the hard one: ease off'];
      return ['bad', 'Too firm: ease off to the centre line'];
    case FCHECK.FAINTER:
      return ['wait', 'A little fainter: under the right edge of the box'];
    case FCHECK.OTHER:
      return ['wait', set.other(arg)];
    case FCHECK.NOT_RELAXED:
      return ['wait', step === FSTEP.TRAIN ? set.letGo(i) : `Waiting for CH${arg + 1} to relax`];
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

function drawCoachBar(set, kind, meter, lo, hi, step, i, clean) {
  let max;
  let zones;
  let label;
  if (kind === FMETER.ACTIVITY) {
    max = 3;
    zones = [[0, 1, 'weak'], [1, 3, 'ok']];
    label = `Strength: ${meter.toFixed(1)}× what's needed`;
  } else if (kind === FMETER.OF_HARD) {
    max = 1;
    zones = [[0, 0.15, 'weak'], [0.15, 0.85, 'ok'], [0.85, 1, 'over']];
    label = `${Math.round(meter * 100)}% of your hard one · aim for the white box`;
  } else if (kind === FMETER.POWER) {
    if (step === FSTEP.FAINT) {
      // zoomed in on the bottom of the range, so a box just above the noise has room
      max = Math.min(POWER_MAX, Math.max(0.25, hi * 2.5));
      zones = [[0, lo, 'weak'], [lo, max, 'ok']];
    } else {
      max = POWER_MAX;
      zones = [[0, lo, 'weak'], [lo, max, 'ok']];
    }
    label = `${set.name(i)}: ${Math.round(meter * 100)}% of the ceiling (${set.ceiling})`
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

function drawCoach(now) {
  if (!coach.active) return;
  const set = SET;
  const s = set.status();
  if (!s.length) {
    finishCalibration();
    return;
  }
  const [step, i, phase, progress, secondsLeft, check, arg, stuck, kind, meter, lo, hi, segment, , round, rounds, , aim, posture, waitFlag] = s;
  const waiting = waitFlag === 1;
  const stalled = isStalled(now);
  logStatus(now, s, stalled);
  const count = Math.ceil(secondsLeft - 1e-3);

  // averaged for display; restarts with every step so it doesn't slide in from the last one
  const phaseKey = `${step}:${i}:${round}:${phase}`;
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
  if (!training) $('trainer-anim').hidden = true;
  if (training) {
    drawTrainer({ set, i, phase, progress, check, round, rounds, lo, hi, aim, shown, stalled });
  } else {
    drawCard({ set, step, i, phase, progress, count, check, arg, stuck, kind, shown, lo, hi, segment, stalled, posture, waiting });
  }

  // sounds on the sample clock: a tone at each switch, ticks before a recording,
  // and for training a chirp on a hit and a low tone on a miss
  // (waiting is in the key so Space starts the countdown with a tick)
  const key = `${step}:${i}:${round}:${phase}:${count}:${waiting}`;
  if (key !== coach.lastKey && !stalled) {
    const [pStep, pI, pRound, pPhase] = coach.lastKey.split(':').map(Number);
    if (pStep !== step || pI !== i || pRound !== round || pPhase !== phase) {
      if (pStep === FSTEP.TRAIN && pPhase === PHASE.RECORD && phase === PHASE.RELAX) {
        if (coach.lastProgress >= 0.9) beep(1250, 130);
        else beep(260, 220, 0.05);
      } else if (phase === PHASE.RECORD) beep(step === FSTEP.REST ? 520 : 880, 180);
      else if (phase === PHASE.RELAX) beep(440, 140);
    } else if (phase === PHASE.PREPARE && !waiting && count <= 3 && step !== FSTEP.TRAIN) {
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
function drawTrainer({ set, i, phase, progress, check, round, rounds, lo, hi, aim, shown, stalled }) {
  const el = $('trainer');
  const active = phase === PHASE.RECORD;
  el.classList.toggle('idle', !active);
  // the gesture being trained, animated at the top of the view
  const anim = $('trainer-anim');
  const pic = set.anim(i);
  anim.hidden = !pic;
  if (pic && !anim.src.endsWith(pic)) {
    anim.src = pic;
    anim.alt = set.name(i);
  }
  $('trainer-what').textContent = `${set.name(i)} → ${jointFor(i)}`;
  $('trainer-trial').textContent = `trial ${round + 1}/${rounds}`;
  $('trainer-skip').textContent = `Skip ${set.noun}`;
  $('trainer-hint').textContent = stalled ? 'Paused: no data from the board'
    : phase === PHASE.PREPARE
      ? 'Next: a light, steady intention, arm still'
      : phase === PHASE.RELAX ? (check === FCHECK.CLEAN ? 'Relaxed' : set.letGo(i))
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

/** Phase 1 (and rest): the full coach card with the power bar, and the posture picture at rest. */
function drawCard({ set, step, i, phase, progress, count, check, arg, stuck, kind, shown, lo, hi, segment, stalled, posture, waiting }) {
  const pose = POSTURES[posture];
  const name = i >= 0 ? set.name(i) : '';
  let cls;
  let label;
  let title;
  let detail;
  if (step === FSTEP.REST) {
    const prep = phase === PHASE.PREPARE;
    [cls, label] = prep ? ['prepare', waiting ? (posture ? 'CHANGE POSITION' : 'GET IN POSITION') : 'GET READY'] : ['rest', 'STAY RELAXED'];
    title = prep ? pose.title : posture ? 'Hold it there, hand relaxed' : 'Stay completely relaxed';
    detail = prep
      ? pose.place
      : posture
        ? `Measuring how this position changes the resting signal, so it isn't read as ${set.restRead}.`
        : 'Measuring the resting noise on every electrode. Only still seconds count.';
  } else if (phase === PHASE.PREPARE) {
    [cls, label] = ['prepare', 'GET READY'];
    title = step === FSTEP.HARD ? `Next: ${name}, hard`
      : step === FSTEP.MEDIUM ? `Next: ${name}, about half as hard` : `Next: ${name}, faint`;
    detail = set.prepare(step, i);
  } else if (phase === PHASE.RECORD) {
    if (step === FSTEP.HARD) {
      [cls, label] = ['real', 'HARD'];
      [title, detail] = set.hard(i);
    } else if (step === FSTEP.MEDIUM) {
      [cls, label] = ['real', 'HALF'];
      title = 'About half as hard';
      detail = 'Still a real contraction, just gentler.';
    } else {
      [cls, label] = ['faint', 'TENSE ONLY'];
      title = 'As faint as you can';
      detail = 'Don\'t move. Barely tense it: keep the line in the box, just past its left edge, '
        + 'which sits just above your resting noise.';
    }
  } else {
    [cls, label] = ['relax', 'RELAX'];
    title = 'Let go';
    detail = 'Waiting for every channel to settle.';
  }
  const stuckLong = phase !== PHASE.PREPARE && stuck >= STUCK_HINT_S && step !== FSTEP.REST;
  if (stuckLong) {
    detail = step === FSTEP.FAINT ? 'Can\'t hold it there? Skip keeps what the half-strength round learned.'
      : `Stuck? Fix the electrode, or skip this ${set.noun}.`;
  }
  if (stalled) detail = 'Paused: no data from the board';

  coach.card.className = `card ${cls}`;
  // phase 1 goes round by round: everything strong, then half, then faint
  const roundName = step === FSTEP.HARD ? 'strong' : step === FSTEP.MEDIUM ? 'half strength' : 'faint';
  const roundNo = step === FSTEP.HARD ? 1 : step === FSTEP.MEDIUM ? 2 : 3;
  $('coach-step').textContent = step === FSTEP.REST
    ? `${set.label} · rest ${posture + 1} of ${POSTURES.length}: ${pose.name}`
    : `Round ${roundNo} of 3 · ${roundName} · ${name}`;
  $('coach-phase').textContent = label;
  $('coach-title').textContent = title;
  $('coach-detail').textContent = detail;
  // the posture at rest, the movement (animated) for a gesture
  const pic = step === FSTEP.REST ? pose.img : i >= 0 ? set.anim(i) : null;
  const img = $('coach-pose');
  img.hidden = !pic;
  if (pic && !img.src.endsWith(pic)) {
    img.src = pic;
    img.alt = title;
  }
  // held until ready, then 3-2-1; while recording only the bar shows progress
  $('coach-ready').hidden = !waiting;
  $('coach-count').textContent = phase === PHASE.PREPARE && !waiting ? count : '';
  $('coach-bar').parentElement.hidden = phase !== PHASE.RECORD;
  $('coach-bar').style.width = `${progress * 100}%`;

  const [checkCls, checkText] = checkMessage(set, check, arg, step, phase, i);
  $('coach-check').className = checkCls;
  $('coach-check').querySelector('.text').textContent = checkText;
  const skip = $('coach-skip');
  // the limp rest is the reference for everything else, so it can't be skipped
  skip.hidden = step === FSTEP.REST && posture === 0;
  skip.textContent = step === FSTEP.REST ? 'Skip this position'
    : step === FSTEP.FAINT ? 'Skip the faint round' : `Skip this ${set.noun}`;
  skip.classList.toggle('urge', stuckLong);

  // the level meter only while it matters: recording or relaxing a gesture
  $('fbar').hidden = step === FSTEP.REST || phase === PHASE.PREPARE;
  if (!$('fbar').hidden) drawCoachBar(set, kind, shown, lo, hi, step, i, check === FCHECK.CLEAN);
  [...$('coach-timeline').children].forEach((d, k) => {
    d.className = k < segment ? 'done' : k === segment ? 'now' : '';
  });
}

// ---- calibration log ---------------------------------------------------------

/** Names for the coach status vector (gesture_status), as logged. */
const STATUS_FIELDS = ['step', 'index', 'phase', 'progress', 'secondsLeft', 'check', 'arg', 'stuck', 'meterKind',
  'meter', 'lo', 'hi', 'segment', 'segments', 'round', 'rounds', 'goal', 'aim', 'posture', 'waiting'];
const LOG_TICK_MS = 100;

/** Every change of step, phase, round or check, and the full status 10× a second. */
function logStatus(now, s, stalled) {
  const status = [...s];
  const key = `${s[0]}:${s[1]}:${s[2]}:${s[14]}:${s[5]}:${s[6]}:${s[19]}:${stalled}`;
  if (key !== coach.lastLogKey) {
    recorder.event('status', { status, stalled, changed: true });
    coach.lastLogKey = key;
    coach.lastLogAt = now;
  } else if (now - coach.lastLogAt >= LOG_TICK_MS) {
    // what the decoder makes of it, and where the arm is, for offline decoders to compare against
    recorder.event('status', {
      status,
      stalled,
      envelopes: [...app.envelopes()],
      contact: [...app.contact()],
      powers: [...app.gesture_powers()],
      active: [...app.gesture_active()],
      posture: [...app.gesture_posture()],
      axisInputs: [...app.gesture_axis_inputs()],
      joints: [...app.joint_angles()],
    });
    coach.lastLogAt = now;
  }
}

/** Everything about a run that the samples alone don't say, for meta.json. */
function runMeta(name, mode, calibrationBefore) {
  const digit = isDigit();
  return {
    name,
    scheme,
    kind: scheme, // older runs said 'gesture' or 'finger'
    mode,
    source: sim ? 'simulator' : 'cyton',
    wiring: wiringFor(),
    sampleRate: 250,
    uvPerCount: 4.5 / 24 / 8388607 * 1e6,
    cytonChannelSettings: 'gain 24, normal input, bias on, SRB1 off, '
      + `SRB2 ${digit ? 'off (bipolar pairs)' : 'on (shared REF)'}`,
    mains: 60,
    car: app.car(),
    filterChain: [
      '20 Hz 4th-order Butterworth high-pass per channel',
      app.car() ? 'common-average reference over channels with contact (3 or more)' : 'no common-average reference',
      'notch at mains (60 Hz) twice and at 120 Hz twice, Q 5',
      'full-wave rectify',
      '5 Hz 2nd-order Butterworth low-pass: the envelope',
    ],
    placement: Array.from({ length: NUM_CH }, (_, ch) => ({
      channel: ch + 1,
      site: digit ? DIGIT_SITES[ch] : 'any (calibration learns each gesture across all channels)',
    })),
    movements: moves().map((m, g) => {
      const [axis, direction] = axesOf()[g];
      return { index: g, name: m.short, cue: m.cue, how: m.how, axis, axisName: AXES[axis], direction, drives: jointFor(g) };
    }),
    axes: AXES,
    postures: POSTURES.map((p) => p.name),
    stepNames: ['rest', 'hard', 'medium', 'faint', 'train'],
    phaseNames: ['prepare', 'record', 'relax'],
    calibrationVersion: app.calibration_version(),
    settings: panelSettings(),
    calibrationBefore,
    statusFields: STATUS_FIELDS,
    tickFields: ['envelopes (µV per channel)', 'contact (per channel)', 'powers (per movement)',
      'active (per movement)', 'posture (match per rest posture)', 'axisInputs (base, reach, elbow, gripper; -1..1)',
      'joints (degrees: base, shoulder, elbow, gripper)'],
    userAgent: navigator.userAgent,
  };
}

/** Writes result.json: how the run ended and everything the decoder learned. */
function recordEnd(outcome) {
  recorder.finish({
    outcome,
    scheme: coach.scheme,
    report: [...app.gesture_report()],
    reportLen: GREPORT_LEN,
    reportFields: ['calibrated', 'faint', 'isolation', 'hits', 'trials', 'level', 'stasis', 'mixed'],
    calibrated: app.gesture_calibrated(),
    bands: [...app.gesture_bands()],
    bandFields: ['stasis', 'restExit', 'level'],
    channelGestures: [...app.channel_gestures()],
    stats: [...app.stats()],
    settings: panelSettings(),
    calibration: toBase64(app.export_calibration()),
  });
}

function finishCalibration() {
  coach.active = false;
  recordEnd('complete');
  saveCalibration();
  $('trainer').hidden = true;
  $('trainer-anim').hidden = true;
  coach.el.hidden = false;
  beep(990, 220);
  coach.card.className = 'card done';
  $('coach-phase').textContent = 'DONE';
  $('coach-title').textContent = 'Levels set';
  finishResults();
  showCoachParts(false);
  refreshRoles();
}

const grade = (v, good, ok) => (v >= good ? 'good' : v >= ok ? 'ok' : 'bad');

/** gesture_report(): [calibrated, faint, isolation, hits, trials, level, stasis, mixed] per gesture. */
const GREPORT_LEN = 8;

function finishResults() {
  const rep = app.gesture_report();
  const noun = SET.noun;
  const rows = moves().map((gs, g) => {
    const [cal, faint, isolation, hits, trials, level, , mixed] = rep.slice(g * GREPORT_LEN, (g + 1) * GREPORT_LEN);
    const name = `<span>${gs.short} → ${jointFor(g)}</span>`;
    const redo = `<button class="small" data-redo="${g}" title="Recalibrate and retrain just this ${noun}">Redo</button>`;
    if (!cal) return `<div class="res gres">${name}<span class="bad">skipped</span><span></span><span></span><span></span>${redo}</div>`;
    const training = trials ? `<span class="${grade(hits / trials, 0.6, 0.4)}">${hits}/${trials} hits</span>` : '<span>—</span>';
    return `<div class="res gres">${name}<span class="mono">${Math.round(faint * 100)}%</span>`
      + `<span class="${mixed ? 'bad' : grade(isolation, 0.9, 0.7)}" title="${mixed ? `The faint round looked more like another ${noun}, so it was not used` : ''}">`
      + `${Math.round(isolation * 100)}%${mixed ? ' · mixed' : ''}</span>${training}`
      + `<span class="mono">${Math.round(level * 100)}%</span>${redo}</div>`;
  });
  $('coach-results').innerHTML = `<div class="res gres head"><span>${cap(noun)} → axis</span><span>faint</span>`
    + '<span>isolated</span><span>training</span><span>level</span><span></span></div>' + rows.join('');
  $('coach-step').textContent = `${SET.label} complete`;
  $('coach-detail').textContent = 'Faint: how far below the ceiling (your half-strength contraction) the faint round read. '
    + `Isolated: how often that faint intention read as the right ${noun} ("mixed": it looked like another one, so it wasn't used). `
    + 'Level: the power that now drives its axis at full speed; training stepped it down after two hits in a row and up after a miss. '
    + `Train only practises again; Redo recalibrates one ${noun}.`;
}

// ---- run name: asked before every calibration run, so recordings can be tracked ----

const runDialog = $('run-name');
const runInput = $('run-name-input');
let afterName = null;

/** Asks for the run's name (prefilled with the last one), then calls then(name). */
function askRunName(what, then) {
  if (!requireStream()) return;
  afterName = then;
  $('run-name-what').textContent = what;
  try { runInput.value = localStorage.getItem('runName') ?? ''; } catch { runInput.value = ''; }
  $('run-name-error').textContent = '';
  runDialog.showModal();
  runInput.select();
}
$('run-name-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const name = runInput.value.trim();
  if (!/[0-9a-z]/i.test(name)) {
    $('run-name-error').textContent = 'Give the run a name (letters or digits), e.g. the person and the session.';
    return;
  }
  try { localStorage.setItem('runName', name); } catch { /* remembering is optional */ }
  const then = afterName;
  afterName = null;
  runDialog.close();
  then(name);
});
$('run-name-cancel').onclick = () => {
  afterName = null;
  runDialog.close();
};

$('gcal').onclick = () => askRunName(`${SET.label}: full`, (name) => {
  showGuide(() => startCalibration(() => app.gesture_calibrate(true), 'full', name));
});
$('gtrain').onclick = () => askRunName(`${SET.label}: train only`, (name) => {
  startCalibration(() => app.gesture_train(), 'train only', name);
});
$('coach-cancel').onclick = () => {
  recorder.event('cancel');
  stopCalibration('cancelled');
};
$('coach-close').onclick = () => { coach.el.hidden = true; };
$('coach-results').addEventListener('click', (e) => {
  const g = e.target.dataset?.redo;
  if (g === undefined) return;
  const what = `redo ${moves()[g].short}`;
  askRunName(`${SET.label}: ${what}`, (name) => startCalibration(() => app.gesture_calibrate_one(Number(g), true), what, name));
});
const skipStep = () => {
  recorder.event('skip');
  SET.skip();
};
$('coach-skip').onclick = skipStep;
$('trainer-skip').onclick = skipStep;
$('trainer-stop').onclick = () => $('coach-cancel').click();
addEventListener('keydown', (e) => {
  if (e.key !== 'Escape' || runDialog.open) return;
  if (coach.active) $('coach-cancel').click();
  else coach.el.hidden = true;
});

// ---- main loop -------------------------------------------------------------

let last = performance.now();
let lastUi = 0;
let lastStats = { t: last, packets: 0, dropped: 0 };

// ---- stream health: packets/s and drops over the last 10 s ----------------------

const STATS_STEP_MS = 500;
const STATS_SLOTS = 20; // × 0.5 s = 10 s
/** The Cyton's rate; the packets chart marks it with a dashed line. */
const EXPECTED_PPS = 250;
const statsHistory = []; // { pps, drops }, both per second
const STREAM_COLORS = { pps: '#33d9ff', drops: '#ff6b6b' };
const streamChart = $('stream-chart');
let statsHover = null; // slot under the pointer, or null

/**
 * Received and dropped packets/s on one scale, newest on the right; empty slots
 * on the left until 10 s have passed. Dropped gets a red fill down to zero.
 */
function drawStats() {
  // the numbers show even when the chart is collapsed
  const offset = STATS_SLOTS - statsHistory.length;
  const i = statsHover !== null && statsHover >= offset ? statsHover - offset : statsHistory.length - 1;
  const s = statsHistory[i];
  $('pps').textContent = Math.round(s?.pps ?? 0);
  $('drops').textContent = Math.round(s?.drops ?? 0);
  const ago = s ? (statsHistory.length - 1 - i) * (STATS_STEP_MS / 1000) : 0;
  $('stream-when').textContent = statsHover !== null && s ? (ago ? `${ago.toFixed(1)} s ago` : 'now') : 'last 10 s';
  const total = statsHistory.reduce((sum, h) => sum + h.drops * (STATS_STEP_MS / 1000), 0);
  $('drops-total').textContent = `${Math.round(total)} dropped in 10 s`;
  if ($('stream').open) drawStreamChart(offset, i, s);
}

function drawStreamChart(offset, i, s) {
  const canvas = streamChart;
  const dpr = Math.min(devicePixelRatio, 2);
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (!w || !h) return;
  if (canvas.width !== w || canvas.height !== h) [canvas.width, canvas.height] = [w, h];
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  const max = Math.max(EXPECTED_PPS * 1.2, ...statsHistory.flatMap((p) => [p.pps, p.drops])) * 1.05;
  const slotW = w / STATS_SLOTS;
  const x = (k) => (offset + k + 0.5) * slotW;
  const y = (v) => h - dpr - (v / max) * (h - 4 * dpr);

  // recessive baseline, and the Cyton's expected rate dashed
  ctx.fillStyle = '#2f333c';
  ctx.fillRect(0, h - dpr, w, dpr);
  ctx.strokeStyle = '#8b919c';
  ctx.lineWidth = dpr;
  ctx.setLineDash([3 * dpr, 3 * dpr]);
  ctx.beginPath();
  ctx.moveTo(0, Math.round(y(EXPECTED_PPS)) + 0.5);
  ctx.lineTo(w, Math.round(y(EXPECTED_PPS)) + 0.5);
  ctx.stroke();
  ctx.setLineDash([]);

  const trace = (key) => {
    ctx.beginPath();
    statsHistory.forEach((p, k) => (k ? ctx.lineTo(x(k), y(p[key])) : ctx.moveTo(x(k), y(p[key]))));
  };
  if (statsHistory.length) {
    // dropped: fill under the line first, so both lines draw over it
    trace('drops');
    ctx.lineTo(x(statsHistory.length - 1), h - dpr);
    ctx.lineTo(x(0), h - dpr);
    ctx.closePath();
    ctx.fillStyle = `${STREAM_COLORS.drops}59`;
    ctx.fill();
    ctx.lineWidth = 2 * dpr;
    ctx.lineJoin = 'round';
    for (const key of ['drops', 'pps']) {
      trace(key);
      ctx.strokeStyle = STREAM_COLORS[key];
      ctx.stroke();
    }
  }

  // hover: a crosshair and a dot on each line at that slot
  if (statsHover !== null && s) {
    ctx.fillStyle = '#e3e6eb88';
    ctx.fillRect(Math.round(x(i)), 0, dpr, h);
    for (const key of ['drops', 'pps']) {
      ctx.fillStyle = STREAM_COLORS[key];
      ctx.beginPath();
      ctx.arc(x(i), y(s[key]), 4 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
$('stream').addEventListener('toggle', drawStats);

streamChart.addEventListener('pointermove', (e) => {
  const r = streamChart.getBoundingClientRect();
  statsHover = Math.max(0, Math.min(STATS_SLOTS - 1, Math.floor(((e.clientX - r.left) / r.width) * STATS_SLOTS)));
  drawStats();
});
streamChart.addEventListener('pointerleave', () => {
  statsHover = null;
  drawStats();
});

function updateUi(now) {
  const angles = app.joint_angles();
  // with a real Cyton streaming, EMG drives the arm joints and their buttons do nothing
  const emgDriven = source === 'emg' && !sim;
  const engaged = Boolean(roarm?.following);
  joints.forEach((j, i) => {
    j.angle.textContent = `${fmt(angles[i])}°`;
    j.real.textContent = engaged && roarm.real ? `real ${fmt(roarm.real[i])}°` : '';
    for (const b of j.buttons) {
      b.classList.toggle('held', held.has(b.dataset.key));
      b.disabled = emgDriven && i !== GRIPPER;
    }
  });

  const env = app.envelopes();
  const act = app.activations();
  const contact = app.contact();
  // every channel is recorded for every pattern in phase 1
  const cal = SET.status();
  const recording = cal.length > 0 && cal[2] === PHASE.RECORD && cal[0] !== FSTEP.TRAIN;
  channels.forEach((c, ch) => {
    c.fill.style.width = `${act[ch] * 100}%`;
    c.uv.textContent = contact[ch] ? `${env[ch].toFixed(1)}µV` : 'off';
    c.row.classList.toggle('nocontact', !contact[ch]);
    c.row.title = contact[ch]
      ? siteName(ch)
      : `${siteName(ch)}: no contact (input railed). Press the electrode down or re-gel it.`;
    c.row.classList.toggle('calibrating', recording);
  });

  const [packets, dropped] = app.stats();
  if (!streaming()) {
    // paused: the charts hold what they last showed, and resume from here on reconnect
    lastStats = { t: now, packets, dropped };
  } else if (now - lastStats.t >= STATS_STEP_MS) {
    // counters restart with each connection, so never a negative step
    const dt = (now - lastStats.t) / 1000;
    statsHistory.push({
      pps: Math.max(0, packets - lastStats.packets) / dt,
      drops: Math.max(0, dropped - lastStats.dropped) / dt,
    });
    if (statsHistory.length > STATS_SLOTS) statsHistory.shift();
    lastStats = { t: now, packets, dropped };
    drawStats();
  }

  drawCalPills();

  const pill = $('mode-pill');
  const uncal = source === 'emg' && !app.gesture_calibrated();
  pill.textContent = source === 'emg' ? (uncal ? 'EMG · uncalibrated' : 'EMG') : 'Manual';
  pill.className = `pill ${source === 'emg' ? (uncal ? 'warn' : 'emg') : ''}`;
}

function frame(now) {
  const dt = (now - last) / 1000;
  last = now;

  if (source === 'manual') {
    for (let j = 0; j < GRIPPER; j++) app.set_input(j, keyAxis(j));
  }
  driveGripper();
  if (sim) pumpSimulator(now);
  trackArm();
  app.frame(dt);
  roarm?.update(app.joint_angles(), dt);
  drawCoach(now);
  drawIntent();

  if (now - lastUi >= UI_INTERVAL_MS) {
    lastUi = now;
    updateUi(now);
    updateArm();
  }
  requestAnimationFrame(frame);
}

// ---- real arm (RoArm-M2-S) ----------------------------------------------------

const setArm = (msg) => { $('arm-status').textContent = msg; };

function updateArmButtons() {
  applyDrift();
  $('arm-connect').textContent = roarm ? 'DISCONNECT ARM' : 'CONNECT ARM';
  $('arm-engage').textContent = roarm?.following ? 'DISENGAGE' : 'ENGAGE';
  $('arm-engage').classList.toggle('active', Boolean(roarm?.following));
  $('arm-engage').disabled = !roarm?.live;
  $('estop').hidden = !roarm;
}

/** E STOP and Space: the real arm stops where it is and is sent nothing more until Engage. */
function emergencyStop() {
  held.clear();
  if (!roarm) return;
  const wasEngaged = roarm.following;
  roarm.halt();
  setArm(wasEngaged ? 'Stopped. The arm holds where it is; press Engage to drive it again.' : 'Stopped. The arm holds where it is.');
  updateArmButtons();
}
$('estop').addEventListener('pointerdown', (e) => {
  e.preventDefault();
  emergencyStop(); // on press, not release
});
$('estop').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') emergencyStop();
});

$('arm-connect').onclick = async () => {
  if (roarm) {
    await roarm.disconnect();
    return;
  }
  if (!RoArm.supported) {
    setArm('Web Serial is unavailable. Use Chrome or Edge on http://localhost.');
    return;
  }
  roarm = new RoArm({
    onStatus: (msg) => { setArm(msg); updateArmButtons(); },
    onClose: () => {
      roarm = null;
      updateArmButtons();
      setArm('');
    },
  });
  updateArmButtons();
  try {
    await roarm.connect();
  } catch (err) {
    const msg = err.name === 'NotFoundError' ? 'No port selected.' : err.message;
    await roarm?.disconnect();
    roarm = null;
    setArm(msg);
  }
  updateArmButtons();
};

$('arm-engage').onclick = () => {
  roarm?.follow(!roarm.following);
  setArm(roarm?.following
    ? "Engaged: the arm is sent the simulation's pose. E STOP or Space stops it."
    : "Disengaged: the simulation shows the arm's pose.");
  updateArmButtons();
};

/** Before each frame: while not following, the simulation takes the real arm's pose. */
function trackArm() {
  if (roarm?.live && !roarm.following) app.track_joints(roarm.real);
}

function updateArm() {
  if (roarm?.servosDown && !roarm.following) setArm(SERVOS_DOWN);
  // following can stop on its own (feedback lost), and Engage needs live feedback
  const engage = $('arm-engage');
  if (engage.classList.contains('active') !== Boolean(roarm?.following) || engage.disabled === Boolean(roarm?.live)) updateArmButtons();
}

// ---- arm meshes ---------------------------------------------------------------
// Waveshare's own link meshes (python fetch_models.py puts them in www/models/);
// until all have loaded the arm is drawn from simple shapes.

async function loadArmMeshes() {
  try {
    const files = await Promise.all(app.link_mesh_names().map(async (name) => {
      const res = await fetch(`models/${name}.stl`);
      if (!res.ok) throw new Error(`models/${name}.stl: ${res.status}`);
      return new Uint8Array(await res.arrayBuffer());
    }));
    files.forEach((bytes, i) => app.load_link_mesh(i, bytes));
  } catch (err) {
    console.info(`Arm drawn from simple shapes (${err.message ?? err}). Run python fetch_models.py for the real meshes.`);
  }
}

// ---- saved calibration -----------------------------------------------------
// serve.py keeps one calibration.json: both schemes' calibrations (WASM bytes,
// base64) plus the panel settings. Saved when a calibration finishes and when a
// setting changes; loaded on startup.

const RANGES = ['gain'];
let autosave = false; // off until the saved file has been read, so it can't be overwritten unread
let saveTimer = null;
const setSaved = (msg) => { $('saved-status').textContent = msg; };

const toBase64 = (bytes) => {
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
};
const fromBase64 = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

function panelSettings() {
  return {
    ...Object.fromEntries(RANGES.map((id) => [id, Number($(id).value)])),
    drift,
  };
}

/** Sets each control and fires its own handler, so the app and the labels follow. */
function applySettings(st) {
  const set = (el, value, event) => {
    if (value === undefined) return;
    if (el.type === 'checkbox') el.checked = value;
    else el.value = String(value);
    el.dispatchEvent(new Event(event, { bubbles: true }));
  };
  for (const id of RANGES) set($(id), st[id], 'input');
  if (typeof st.drift === 'boolean') setDrift(st.drift);
}

async function saveCalibration() {
  clearTimeout(saveTimer);
  autosave = true; // a finished calibration is worth keeping even if the old file was unreadable
  const body = JSON.stringify({ settings: panelSettings(), calibration: toBase64(app.export_calibration()) });
  try {
    const res = await fetch('/calibration', { method: 'PUT', body });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    setSaved('');
  } catch (err) {
    setSaved(`Couldn't save the calibration: ${err.message}`);
  }
}

function saveSoon() {
  if (!autosave || coach.active) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveCalibration, 800);
}
$('app').addEventListener('change', saveSoon);
$('app').addEventListener('input', saveSoon);

async function loadCalibration() {
  let saved;
  try {
    const res = await fetch('/calibration');
    if (res.status === 404) {
      autosave = true;
      return;
    }
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    saved = await res.json();
    app.import_calibration(fromBase64(saved.calibration));
  } catch (err) {
    setSaved(`Saved calibration not loaded (${err.message ?? err}). The file is kept until a new calibration finishes.`);
    return;
  }
  applySettings(saved.settings ?? {});
  refreshRoles();
  autosave = true;
}

setSource('manual');
let savedScheme = 'muscle';
try { savedScheme = localStorage.getItem('scheme') === 'digit' ? 'digit' : 'muscle'; } catch { /* default */ }
setScheme(savedScheme);
updateButtons();
loadCalibration();
loadArmMeshes();
requestAnimationFrame(frame);
