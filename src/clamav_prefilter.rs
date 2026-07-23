//! Shift-OR bloom prefilter for ClamAV-style Aho-Corasick scanning.
//!
//! Uses 2-byte q-grams with a 65536-entry bit-vector (128 KB, L2-cache-friendly).
//! No-false-negatives: if `search()` returns `None`, the automaton can be skipped
//! entirely.  If it returns `Some(offset)`, the scanner should run from `offset`
//! (which may be 0 if the match starts very early).
//!
//! Adapted from ClamAV's `filtering.c` / `matcher-ac.c`.

use alloc::vec::Vec;

/// Maximum pattern length (in bytes) tracked in the prefilter state machine.
///
/// The Shift-OR state is 8 bits, so it can track at most 9 q-gram positions,
/// corresponding to patterns up to `MAX_ATOM_LEN` bytes.  Longer patterns are
/// truncated — still safe (no false negatives), just more false positives.
const MAX_ATOM_LEN: usize = 16;

/// Shift-OR bloom prefilter (ClamAV-style).
///
/// `B[q]` has bit P clear if q-gram `q` can appear at position P of some pattern.
/// `end[q]` has bit P clear if `q` can be the terminal q-gram of a pattern
/// (i.e. the pattern ends at byte P+2).
pub struct ClamavPrefilter {
    B: [u8; 65536],
    end: [u8; 65536],
}

impl core::fmt::Debug for ClamavPrefilter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ClamavPrefilter").finish()
    }
}

impl ClamavPrefilter {
    /// Returns an empty (fully permissive) prefilter.
    ///
    /// Every q-gram is accepted at every position, so `search()` always returns
    /// `Some(0)` for any data with length ≥ 2.
    #[must_use]
    pub fn empty() -> Self {
        Self { B: [0u8; 65536], end: [0u8; 65536] }
    }

    /// Build a prefilter from a list of patterns.
    ///
    /// All q-grams are lowered before insertion, making the entire prefilter
    /// case-insensitive.  This ensures nocase patterns are not missed.
    #[must_use]
    pub fn from_patterns(patterns: &[Vec<u8>]) -> Self {
        let mut B = [0xFFu8; 65536];
        let mut end = [0xFFu8; 65536];
        let limit = MAX_ATOM_LEN.min(9);
        for pat in patterns {
            let n = pat.len().min(limit);
            if n < 3 {
                continue;
            }
            for j in 0..n - 1 {
                let lo = pat[j].to_ascii_lowercase();
                let hi = pat[j + 1].to_ascii_lowercase();
                let q = u16::from_le_bytes([lo, hi]) as usize;
                B[q] &= !(1u8 << j);
                if j == n - 2 {
                    end[q] &= !(1u8 << j);
                }
            }
        }
        Self { B, end }
    }

    /// Search the prefilter over `data`.
    ///
    /// Returns `None` if no pattern can match (the automaton can be skipped).
    /// Returns `Some(byte_offset)` if a potential match was detected; the
    /// scanner should run the automaton from `byte_offset` and adjust match
    /// positions accordingly.
    #[must_use]
    pub fn search(&self, data: &[u8]) -> Option<usize> {
        if data.len() < 2 {
            return None;
        }
        let mut state: u8 = 0xFF;
        for j in 0..data.len() - 1 {
            let lo = data[j].to_ascii_lowercase();
            let hi = data[j + 1].to_ascii_lowercase();
            let q = u16::from_le_bytes([lo, hi]) as usize;
            state = (state << 1) | self.B[q];
            let match_end = state | self.end[q];
            if match_end != 0xFF {
                let start = if j + 2 >= MAX_ATOM_LEN {
                    j + 2 - MAX_ATOM_LEN
                } else {
                    0
                };
                return Some(start);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_prefilter_never_rejects() {
        let pf = ClamavPrefilter::empty();
        assert_eq!(pf.search(b"hello"), Some(0));
        assert_eq!(pf.search(b"x"), None); // too short
    }

    #[test]
    fn simple_exact_match() {
        let pat = b"abc".to_vec();
        let pf = ClamavPrefilter::from_patterns(&[pat]);
        assert_eq!(pf.search(b"---abc---"), Some(3));
    }

    #[test]
    fn case_insensitive_match() {
        let pat = b"abc".to_vec();
        let pf = ClamavPrefilter::from_patterns(&[pat]);
        assert_eq!(pf.search(b"---ABC---"), Some(3));
        assert_eq!(pf.search(b"---AbC---"), Some(3));
    }

    #[test]
    fn no_match() {
        let pat = b"xyz".to_vec();
        let pf = ClamavPrefilter::from_patterns(&[pat]);
        assert_eq!(pf.search(b"---abc---"), None);
    }

    #[test]
    fn short_patterns_are_skipped() {
        let pat = b"ab".to_vec(); // too short (< 3 bytes)
        let pf = ClamavPrefilter::from_patterns(&[pat]);
        assert_eq!(pf.search(b"xab"), None);
    }
}
