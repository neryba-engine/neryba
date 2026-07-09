//! Zobrist hashing (ADR-0004 baseline). Keys from a splitmix64 stream with a
//! fixed seed — deterministic, no dependencies.

const fn splitmix(state: u64) -> (u64, u64) {
    let s = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    (z ^ (z >> 31), s)
}

const fn build_keys<const N: usize>(mut seed: u64) -> [u64; N] {
    let mut out = [0u64; N];
    let mut i = 0;
    while i < N {
        let (k, s) = splitmix(seed);
        out[i] = k;
        seed = s;
        i += 1;
    }
    out
}

/// [color 2][ptype 1..=6 -> idx 0..6][square 64]; flattened.
pub const PIECE_KEYS: [u64; 2 * 6 * 64] = build_keys(0x0BAD_F00D_2026_0703);
pub const CASTLE_KEYS: [u64; 16] = build_keys(0x5EED_CA57);
pub const EP_FILE_KEYS: [u64; 8] = build_keys(0x5EED_E9A5);
pub const STM_KEY: u64 = 0x9E3779B97F4A7C15;

#[inline]
pub fn piece_key(color: u8, pt: u8, sq: u8) -> u64 {
    PIECE_KEYS[((color as usize) * 6 + (pt as usize - 1)) * 64 + sq as usize]
}
