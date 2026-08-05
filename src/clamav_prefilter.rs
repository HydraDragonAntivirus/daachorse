//! Shift-OR bloom prefilter for ClamAV-style Aho-Corasick scanning.
//!
//! Uses 2-byte q-grams with a 65536-entry bit-vector (128 KB, L2-cache-friendly).
//! No-false-negatives: if `search()` returns `None`, the automaton can be skipped
//! entirely.  If it returns `Some(offset)`, the scanner should run from `offset`
//! (which may be 0 if the match starts very early).
//!
//! Adapted from ClamAV's `filtering.c` / `matcher-ac.c`.

use alloc::boxed::Box;
use alloc::vec::Vec;

/// Shift-OR bloom prefilter (ClamAV-style).
///
/// `b[q]` has bit P clear if q-gram `q` can appear at position P of some pattern.
/// `end[q]` has bit P clear if `q` can be the terminal q-gram of a pattern
/// (i.e. the pattern ends at byte P+2).  Only exact-case q-grams are tracked
/// (no lowering); with very large pattern sets the filter can saturate.
#[derive(Clone)]
pub struct ClamavPrefilter {
    b: Box<[u8; 65536]>,
    end: Box<[u8; 65536]>,
}

impl core::fmt::Debug for ClamavPrefilter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ClamavPrefilter").finish()
    }
}

impl ClamavPrefilter {
    /// Empty prefilter that never rejects data.
    #[must_use]
    pub fn empty() -> Self {
        Self { b: Box::new([0u8; 65536]), end: Box::new([0u8; 65536]) }
    }

    /// Build from raw bit-vectors (for deserialisation).
    #[must_use]
    pub fn from_raw(b: [u8; 65536], end: [u8; 65536]) -> Self {
        Self { b: Box::new(b), end: Box::new(end) }
    }

    /// Expose the `b` table for serialisation.
    #[must_use]
    pub fn raw_b(&self) -> &[u8; 65536] { &self.b }

    /// Expose the `end` table for serialisation.
    #[must_use]
    pub fn raw_end(&self) -> &[u8; 65536] { &self.end }

    /// Build from exact-case patterns.  Patterns shorter than 3 bytes are
    /// skipped (they can't form a 2-byte q-gram).
    #[must_use]
    pub fn from_patterns(patterns: &[Vec<u8>]) -> Self {
        let mut b = Box::new([0xFFu8; 65536]);
        let mut end = Box::new([0xFFu8; 65536]);
        for pat in patterns {
            let n = pat.len().min(9);
            if n < 3 { continue; }
            for j in 0..n - 1 {
                let q = u16::from_le_bytes([pat[j], pat[j + 1]]) as usize;
                b[q] &= !(1u8 << j);
                if j == n - 2 {
                    end[q] &= !(1u8 << j);
                }
            }
        }
        Self { b, end }
    }

    /// Search the prefilter over `data`.
    ///
    /// Returns `None` if no pattern can match (skip the automaton).
    /// Returns `Some(byte_offset)` if a potential match was detected.
    #[must_use]
    pub fn search(&self, data: &[u8]) -> Option<usize> {
        if data.len() < 2 { return None; }
        let mut state: u8 = 0xFF;
        for j in 0..data.len() - 1 {
            let q = u16::from_le_bytes([data[j], data[j + 1]]) as usize;
            state = (state << 1) | self.b[q];
            if (state | self.end[q]) != 0xFF {
                let start = if j + 2 >= 16 { j + 2 - 16 } else { 0 };
                return Some(start);
            }
        }
        None
    }

    /// Returns `true` if this filter has no patterns registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.b.iter().all(|&x| x == 0xFF)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_accepts_everything() {
        let pf = ClamavPrefilter::empty();
        assert_eq!(pf.search(b"hello"), Some(0));
        assert_eq!(pf.search(b"x"), None);
    }

    #[test]
    fn simple_match() {
        let pf = ClamavPrefilter::from_patterns(&[b"abc".to_vec()]);
        assert_eq!(pf.search(b"---abc---"), Some(0));
    }

    #[test]
    fn no_match() {
        let pf = ClamavPrefilter::from_patterns(&[b"xyz".to_vec()]);
        assert_eq!(pf.search(b"---abc---"), None);
    }

    #[test]
    fn short_patterns_skipped() {
        let pf = ClamavPrefilter::from_patterns(&[b"ab".to_vec()]);
        assert_eq!(pf.search(b"xab"), None);
    }
}
