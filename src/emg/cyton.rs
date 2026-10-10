//! Streaming parser for the OpenBCI Cyton binary format (8 channels, 250 Hz).
//!
//! Packet layout (33 bytes):
//!   [0]      0xA0 header
//!   [1]      sample number (wraps at 255)
//!   [2..26]  8 × 24-bit big-endian two's-complement EXG samples
//!   [26..32] 6 bytes aux / accelerometer
//!   [32]     footer 0xC0..=0xC6

pub const PACKET_LEN: usize = 33;
pub const NUM_CHANNELS: usize = 8;
pub const SAMPLE_RATE: f32 = 250.0;
const HEADER: u8 = 0xA0;
/// Default ADS1299 gain on the Cyton (and what serial.js configures).
pub const GAIN: f32 = 24.0;
pub const UV_PER_COUNT: f32 = 4.5 / GAIN / 8_388_607.0 * 1e6;

pub type Sample = [f32; NUM_CHANNELS];

pub struct CytonParser {
    buf: [u8; PACKET_LEN],
    len: usize,
    last_seq: Option<u8>,
    pub packets: u64,
    pub dropped: u64,
    pub resyncs: u64,
}

impl CytonParser {
    pub fn new() -> Self {
        Self { buf: [0; PACKET_LEN], len: 0, last_seq: None, packets: 0, dropped: 0, resyncs: 0 }
    }

    /// Forget partial packets and sequence history (e.g. after reconnecting).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Feed raw serial bytes; calls `on_sample` with µV values for every complete packet.
    pub fn push(&mut self, bytes: &[u8], mut on_sample: impl FnMut(&Sample)) {
        for &b in bytes {
            if self.len == 0 && b != HEADER {
                continue;
            }
            self.buf[self.len] = b;
            self.len += 1;
            if self.len < PACKET_LEN {
                continue;
            }
            if (0xC0..=0xC6).contains(&b) {
                self.len = 0;
                on_sample(&self.decode());
            } else {
                self.resync();
            }
        }
    }

    fn decode(&mut self) -> Sample {
        let seq = self.buf[1];
        if let Some(prev) = self.last_seq {
            self.dropped += seq.wrapping_sub(prev.wrapping_add(1)) as u64;
        }
        self.last_seq = Some(seq);
        self.packets += 1;
        std::array::from_fn(|ch| {
            let i = 2 + ch * 3;
            let raw = i32::from_be_bytes([0, self.buf[i], self.buf[i + 1], self.buf[i + 2]]);
            let raw = (raw << 8) >> 8; // sign-extend 24 -> 32 bits
            raw as f32 * UV_PER_COUNT
        })
    }

    /// Bad footer: restart from the next header byte inside the buffer, if any.
    fn resync(&mut self) {
        self.resyncs += 1;
        match self.buf[1..].iter().position(|&b| b == HEADER) {
            Some(p) => {
                let start = p + 1;
                self.buf.copy_within(start.., 0);
                self.len = PACKET_LEN - start;
            }
            None => self.len = 0,
        }
    }
}

#[cfg(test)]
pub(crate) fn encode_packet(seq: u8, counts: [i32; NUM_CHANNELS]) -> [u8; PACKET_LEN] {
    let mut p = [0u8; PACKET_LEN];
    p[0] = HEADER;
    p[1] = seq;
    for (ch, c) in counts.iter().enumerate() {
        let b = c.to_be_bytes();
        p[2 + ch * 3..5 + ch * 3].copy_from_slice(&b[1..4]);
    }
    p[32] = 0xC0;
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(parser: &mut CytonParser, bytes: &[u8]) -> Vec<Sample> {
        let mut out = Vec::new();
        parser.push(bytes, |s| out.push(*s));
        out
    }

    #[test]
    fn decodes_sign_and_scale() {
        let mut p = CytonParser::new();
        let pkt = encode_packet(0, [1, -1, 8_388_607, -8_388_608, 0, 1000, -1000, 42]);
        let s = collect(&mut p, &pkt);
        assert_eq!(s.len(), 1);
        assert!((s[0][0] - UV_PER_COUNT).abs() < 1e-9);
        assert!((s[0][1] + UV_PER_COUNT).abs() < 1e-9);
        assert!((s[0][2] - 187_500.0).abs() < 1.0); // +full scale = 4.5V/24
        assert!(s[0][3] < -187_500.0);
        assert!((s[0][5] - 1000.0 * UV_PER_COUNT).abs() < 1e-4);
    }

    #[test]
    fn handles_split_chunks_garbage_and_drops() {
        let mut p = CytonParser::new();
        let mut stream = b"OpenBCI V3 8-16 channel$$$".to_vec();
        stream.extend_from_slice(&encode_packet(10, [5; 8]));
        stream.extend_from_slice(&encode_packet(11, [6; 8]));
        stream.extend_from_slice(&encode_packet(14, [7; 8])); // 2 dropped
        let mut n = 0;
        for chunk in stream.chunks(7) {
            n += collect(&mut p, chunk).len();
        }
        assert_eq!(n, 3);
        assert_eq!(p.dropped, 2);
    }

    #[test]
    fn resyncs_after_truncated_packet() {
        let mut p = CytonParser::new();
        let good = encode_packet(1, [9; 8]);
        let mut stream = good[..20].to_vec(); // truncated packet
        stream.extend_from_slice(&encode_packet(2, [9; 8]));
        stream.extend_from_slice(&encode_packet(3, [9; 8]));
        let s = collect(&mut p, &stream);
        assert!(s.len() >= 1, "should recover at least the last packet");
        assert!(p.resyncs >= 1);
        assert!(s.iter().all(|x| (x[0] - 9.0 * UV_PER_COUNT).abs() < 1e-6));
    }
}
