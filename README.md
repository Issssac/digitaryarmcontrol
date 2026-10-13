# EMG Arm

Drive a Waveshare RoArm-M2-S desktop arm, simulated and optionally the real one,
with forearm/upper-arm muscle signals from an OpenBCI Cyton. The simulated arm uses
Waveshare's own joint chain and link meshes, so its four joints mean exactly what
the real arm's do: the base turns (left/right), the shoulder leans the upper arm
forward and back, the elbow bends the forearm, and the fourth joint opens and
closes the gripper. Kinematics, rendering (WebGL2) and the EMG signal chain are Rust
compiled to WASM; the page talks to the Cyton dongle directly via Web Serial.

## Build & run

One-time setup (Windows):

```powershell
winget install Rustlang.Rustup
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

Build and serve:

```powershell
wasm-pack build --target web --out-dir www/pkg --no-typescript
python fetch_models.py   # once: Waveshare's arm meshes (MIT) into www/models/
python serve.py          # serves www/ on :8080 with caching disabled
```

Without the meshes the arm is drawn from simple shapes; everything else works the same.

Open <http://localhost:8080> in **Chrome or Edge** (Web Serial needs a Chromium
browser and a secure context; `localhost` counts).

Tests (parser, filters, pipeline, kinematics): `cargo test`

### WiFi Shield instead of the dongle

Set **Link → WiFi Shield** and enter the shield's IP before clicking
Connect. The page has to be served by `serve.py`, which bridges the shield: it
proxies the shield's HTTP API (`/shield/api/...`, since the shield sends no CORS
headers) and runs the TCP server on port **3000** that the shield streams raw
packets to, relaying them to the page (`/shield/stream`).

- In AP mode, join the shield's `OpenBCI-xxxx` network; the shield is at `192.168.4.1`.
  On your own WiFi, use the IP your router gave it.
- The shield switches the Cyton to 1000 Hz; Connect sets it back to 250 Hz (`~6`)
  along with the usual channel settings.
- **"nothing reached …:3000"**: allow Python through Windows Firewall on private
  networks (inbound TCP 3000).

## Using it

- **Manual**: Q/A base, W/S shoulder, E/D elbow, or drag the sliders.
- **Gripper**: a button, not a joint you steer: **Open/Close** in the Joints panel, or
  R open, F close, Space toggles, in any control mode. It moves to fully open (60°) or
  closed by itself, and EMG and drift leave it alone. To pick up the box, open it,
  bring the tool point to the box and close it; opening it again drops the box. The arm won't lower the tool point into the desk.
  Drag the 3D view to orbit, scroll to zoom.
- **Simulate**: generates a fake Cyton byte stream (DC offsets, 60 Hz hum, noise)
  that goes through the real parser and filters. Holding a joint key "contracts"
  the channel mapped to it, so you can try calibration and mapping with no hardware.
- **Connect**: over WiFi, talks to the shield at the given IP; over USB, pick the dongle's COM port. The page resets the board, configures all 8
  channels for EMG (gain 24, bias on, SRB2 per the **Electrodes** setting) and starts streaming.

### First session with electrodes

1. Plug in the dongle (switch on **GPIO6**), turn the Cyton on (**PC** position).
2. Connect, then check that **Packets/s ≈ 250** and the channel bars sit near zero at rest.
3. Click **Full calibration** (about a minute). A guide over the 3D view walks
   through each step with a countdown, a progress bar, a live level meter and
   audio cues: first relax completely (rest floor), then for each muscle **GET
   READY → PULSE → RELAX**. A pulse is light *intention*: tense as if about to
   make the movement, but keep the arm still. That level becomes full speed, so
   later pulses drive the arm without real movement. The schedule runs on the
   board's sample clock, so the cue and the recorded window always line up.
   Each phase shows a live signal check, and that check decides when the phase
   advances:
   - **Rest:** counts only still seconds. All electrodes need contact, and a jump
     on any channel reads as movement.
   - **Pulse:** counts only seconds where the cued muscle is at least 2× its rest.
   - **Relax:** waits until every channel is back near rest.

   The countdown pauses while the signal isn't clean, and only clean samples are
   recorded. A dead electrode can be passed with **Skip this step**. At
   the end each muscle gets a pulse-to-rest ratio; redo a weak one with its
   **Pulse** button.
4. Tune **Threshold** (dead-zone) and **Gain**. Pulsing the + muscle of a pair
   moves the joint one way, the − muscle the other; co-contraction cancels.
   Joints use hysteresis: they start at the threshold and stop at half of it.

## Finger grid mode

**Control scheme → Finger grid** swaps muscle pairs for 8 electrodes in a 2×4 grid
on the underside of the forearm, which is usually hairless (diagram: `www/finger-grid.svg`,
also under **Electrode guide**).

- **Grid placement:**
  - Upper row (CH1–4): a quarter of the way from the elbow crease to the wrist.
  - Middle row (CH5–8): halfway down.
  - Each row runs from the thumb side to the pinky side, with gel centres about 2 cm apart.
  - The order within a row (or even across rows) doesn't matter. Nothing in the decoder
    is tied to a position: calibration learns each finger's pattern from whatever site
    is on which channel. Recalibrate after moving electrodes.
  - With large pads (e.g. 3M Red Dot), trim or overlap the foam, but the gel spots must
    not touch. Tape over the grid.
- **Wiring:** either option works.
  - Single row (10 pads): REF on the elbow point.
  - Pairs (17 pads): two pads per site along the forearm.
- **Each finger drives one joint.** Defaults (change them in the panel):
  - Thumb → base
  - Index → shoulder
  - Middle → elbow
  - Ring + little → off
  - **Off** takes a finger out of the control: it drives nothing and is skipped in
    training. It's still calibrated, since the fit needs every finger's pattern to tell
    the others apart.
- **Simple bands (the default for now):** power under ¼ of the ceiling holds, ¼ to ½
  drives −, and over ½ drives + with no upper cap. There's a little hysteresis (back to
  rest under 0.2, back to − under 0.45), and real movements are *not* rejected. Turn
  **Simple bands** off in the panel for the adaptive bands below.
- **Adaptive bands:** how firmly you tense the finger picks the direction. Power runs
  from 0 (rest) to 1 (the ceiling, where the finger really starts to move):

  | Power | Joint |
  |---|---|
  | rest, plus a buffer | holds |
  | faint (around the finger's level) | − direction, once the level has settled for 0.15 s, so a rise to + never twitches − |
  | firmer (above the split) | + direction, after 0.1 s above it, so overshooting on the way to faint doesn't flick + |
  | above the ceiling | every joint holds: that's a real hand movement, not a command |

- **The stasis band** sits just above your measured resting noise (its 99th percentile
  plus a small margin, at least 4% of the ceiling).
- **Arm posture:** raising the forearm tenses the flexors under the grid a little, and
  without help that resting tone reads as finger tension.
  - Calibration records rest in three postures: limp on the table, forearm raised with
    the elbow down, and arm held up unsupported. Each raised posture becomes a pattern
    of its own.
  - A slow fit (averaged over 0.5 s, with the finger patterns in it too) works out
    which posture the arm is in. That posture's baseline is taken out before the finger
    powers are measured.
  - Each finger's noise floor blends between the floors measured in each posture.
  - The panel shows the posture the decoder sees (e.g. "forearm raised 89%").
  - Raised postures can be skipped; the limp one can't, since everything is measured
    against it.
- **Decoding:** each finger's power comes from a non-negative fit of all four finger
  patterns across the grid. A finger leaking onto its neighbours' electrodes doesn't
  move their joints, and electrodes that lose contact are left out of the fit.

**Calibrate + train** (about 4–5 min) runs two phases. All meters are averaged over
about 0.25 s so they don't jitter.

1. **Phase 1, round by round** (full-screen guide with a power bar). Each round goes
   through every finger before the next round starts:
   - **Round 1, strong flex** (a real movement): learns what each finger looks like.
   - **Round 2, half strength:** sets each finger's ceiling (the softest part of that
     flex). Every finger's strong pattern is known by now, so if a different finger
     is the one lighting up, the guide names it ("Index is stronger: keep the other
     fingers loose") and doesn't record.
   - **Round 3, faint:** tension only, in a box from just above your resting noise
     up to 30% of the ceiling. The same wrong-finger check applies. The round
     re-learns the finger's pattern only if it still looks more like that finger
     than any other; otherwise the results mark it "mixed" and leave the pattern alone.
2. **Phase 2, per finger** (a small meter over the 3D view, so the arm stays visible):
   - 8 trials alternate faint (−) and firmer (+).
   - The meter's centre line is the target level. The fill grows red to the right
     when you're over it and green to the left when you're under; each end is the
     edge of that band.
   - Bring it to the centre and hold for a second, while only that finger's joint moves.
   - **The targets stay put; the decoder learns you.** It watches where your faint
     and firm attempts really average (raw power over each trial's last second) and
     remaps that finger's power axis, so your typical faint lands mid −-band and
     your typical firm mid +-band.
     - The remap moves halfway toward what it saw the first time, then ⅓, ¼, ⅕ …,
       so it settles on your average rather than chasing you.
     - It carries over between **Train only** sessions until you recalibrate.
   - **Never ambiguous:** the remap keeps your faint and firm at least twice your
     own wobble apart (and at least 60% of the faint level), and it never stretches
     resting noise (1.3× the noise floor) over the rest line.
   - **Wobble filter:** the more fast jitter you have on a finger relative to the
     band width, the more that finger is smoothed. That runs from 60 ms up to at
     most 200 ms of extra lag. Slow drift doesn't count, since smoothing can't remove it.
   - The results show where you held each direction ("you held") and the smoothing
     each finger got.
   - (Adaptive bands) both targets always fit under the ceiling. In a noisy posture
     or on a weak finger they squeeze together, but the + band never disappears.

The results list each finger's faintest level, how often that faint tension read as
the right finger ("isolated"), training hits, and the final faint level.
**Redo** on a finger's row recalibrates just that finger. **Train only** practises
again.

**Drift home when idle** (Joints panel, on by default): under EMG control, once no
joint has had a command for 1.5 s, the arm eases back to the home pose at up to 6°/s.
A joint you're holding still doesn't drift while you drive another one, and nothing
drifts while the real arm is following.

## Real arm (Waveshare RoArm-M2-S)

The simulation can drive a RoArm-M2-S plugged into this computer by USB, with the
same JSON serial commands as
[roarm-haptic-bridge](https://github.com/maxwellarmstrong/roarm-haptic-bridge)
(`www/roarm.js`). Close that repo's dashboard first: only one program can hold the port.

The two share joint angles 1:1, and the panel lists both side by side.

1. **Real arm → Connect arm**, pick the RoArm's COM port (not the Cyton dongle).
   The simulated arm jumps to the real arm's pose and keeps mirroring it: move the
   real arm by hand and the simulation follows. Keys and EMG don't move anything yet.
2. Tick **Follow the simulation** to switch over. The simulation starts from the real
   arm's pose, so nothing jumps, and from then on the real arm is sent the
   simulation's pose (manual keys or EMG). Untick it to go back to mirroring.

Targets are clamped to the arm's limits (base ±180°, shoulder ±90°, elbow −20…169°,
gripper −10…90°) and move at most 60°/s. Unticking Follow, disconnecting, or the
arm going silent for 1 s stops the commands, and the arm holds where it is.

## Saved calibration

Both schemes' calibrations and the panel settings (mapping, threshold, gain, speed,
finger → joint, bands, decoder, mains, drift) are saved to `calibration.json` next
to `serve.py` whenever a calibration finishes or a setting changes, and loaded
when the page opens. Delete the file to start over. It's per person and per
electrode placement, so recalibrate after moving electrodes.

## Electrode placement (default mapping)

![Electrode placement and Cyton wiring](www/electrodes.svg)

The same diagram is in the app under **OpenBCI Cyton → Electrode guide**.

The Cyton's silkscreen labels both header rows N1P…N8P. The **bottom** row is each
channel's negative input, the **top** row its positive input. Pick the matching
**Electrodes** option in the app before clicking Connect:

- **Single row · shared REF** (default, standard 10-lead cable): one electrode on each
  muscle belly → **bottom** pin of its channel; **REF** on the elbow point (olecranon)
  → **bottom SRB** pin; **BIAS** on the wrist bone (ulnar styloid) → **bottom BIAS**
  pin. Top row stays empty. The page turns SRB2 on (`x n 060110X`).
- **Pairs · top + bottom** (if you have leads for both rows): two electrodes ~2 cm
  apart along each muscle → top and bottom pin of the same channel; BIAS on bone;
  SRB unconnected. SRB2 off (`x n 060100X`). Pairs reject crosstalk from
  neighbouring muscles better.

| Joint     | + channel                | − channel                    |
|-----------|--------------------------|------------------------------|
| Base      | CH1 pronator teres       | CH2 brachioradialis          |
| Shoulder  | CH3 anterior deltoid     | CH4 posterior deltoid        |
| Elbow     | CH5 biceps               | CH6 triceps                  |

You can reassign any channel to any joint direction in the **Mapping** table.

## Signal chain (per channel, 250 Hz)

Raw µV → 4th-order Butterworth high-pass 20 Hz → **common-average reference**
(single-row mode: subtract the mean of the channels with contact, cancelling the
hum and noise the shared REF puts on all of them) → wide notches (Q 5, each applied
twice) at mains and its 2nd harmonic → full-wave rectify → 5 Hz low-pass (envelope)
→ normalise between the rest floor and the pulse level (90th percentiles of
each recording) → `input = hysteresis(a+ − a−, threshold) × gain` → joint velocity.

Channels whose input is railed (electrode off) are shown as **off**, left out of
the common average, and read as zero.

### Decoder: pattern vs. per channel

- **Pattern · all channels** (default once calibrated): each sample is explained
  as a non-negative mix of the calibrated movement patterns (synergy
  decomposition by NNLS).
  - Channels are scaled by their resting noise, and offline electrodes are left
    out of the fit.
  - Each movement gets its own intensity (1 = the calibrated pulse), and
    crosstalk is part of each pattern instead of reading as another movement.
  - **Compound movements:** up to two movements drive their joints at once. The
    weaker one only counts if it reaches 35% of the stronger; below that it's
    treated as crosstalk.
  - Opposite movements on one joint net out. A pattern split between, say,
    wrist flexors and extensors gives a small net movement, not the wrong one.
  - Each movement engages at the threshold and holds down to half of it.
  - The **Mapping** table decides which movement drives which joint direction.
  - The overlay over the 3D view shows the movements in play and their intensity.
  - The calibration results list how often each movement was recognised on
    held-out data, and which ones get mixed up.
- **Per channel**: each joint follows its own pair of channels, as before.

**Offline electrodes:** once calibrated, a channel that loses contact is
estimated from the working ones, and shown with a `~` and a dashed bar.
1. The working channels give each movement's probability.
2. The missing channel is then predicted from how it varied with the others,
   within each movement, during calibration.

Both decoders keep running on the estimate. Calibration never uses estimates.

Why: on a single-row setup the 60 Hz hum came in at about 290 µV, identical on
every channel. The board's clock puts it at about 59.9 Hz, where a narrow notch
barely works. The common average plus wide notches brought the resting envelope
from about 42 µV down to 2–4 µV.

## Troubleshooting

- **"No response from the board"**: wrong COM port, or the dongle switch is on RESET.
- **"Dongle answered but the board did not"**: Cyton off or out of range; try
  power-cycling both.
- **Data lags about half a second, or arrives in bursts / choppy**: set the FTDI latency
  timer to 1 ms (Device Manager → COM port → Port Settings → Advanced), then replug the
  dongle. The Windows default of 16 ms never expires on a continuous stream, so the driver
  waits for a full 4 KB transfer, which is about 480 ms of Cyton data. The setting is per
  dongle, so a new dongle needs it again.
- **Bars high at rest**: check skin prep/gel and that BIAS (and REF in single-row
  mode) are attached on bone; recalibrate rest.
- **Board never answers / BOARD_NOT_READY in the OpenBCI GUI, but the dongle reports
  "System is Up"**: the Cyton's main firmware is missing or non-standard. Reflash
  the official Cyton firmware (v3.1.2) through the dongle. To enter the bootloader,
  hold PROG while switching the board OFF → PC; the blue LED blinks.
- **Mains**: switch to 50 Hz outside North America.

## Layout

```
src/arm.rs            RoArm-M2-S joint chain (Waveshare URDF), limits, velocity integration, desk stop
src/world.rs          desk and box: open/close the gripper on it to pick it up
src/render/           WebGL2 renderer, meshes, orbit camera, arm → parts
src/emg/cyton.rs      33-byte packet parser (resync, drop counting, µV scaling)
src/emg/filter.rs     biquads
src/emg/pipeline.rs   envelope, guided calibration, decoder / channel-pair → joint commands
src/emg/decoder.rs    synergy (NNLS) decoder, offline-channel imputer, cross-validation report
src/emg/finger.rs     finger grid: per-finger power, rest/faint/firmer bands, phase 1 + 2 coach
src/lib.rs            wasm-bindgen API used by the page
www/                  index.html, main.js (UI), serial.js (Web Serial), wifi.js (WiFi Shield),
                      roarm.js (real arm), simulator.js
```
