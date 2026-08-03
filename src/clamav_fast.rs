//! Self-contained ClamAV-style dense-DFA Aho-Corasick scanner.
//!
//! Unlike [`DoubleArrayAhoCorasick`](crate::DoubleArrayAhoCorasick), this
//! structure owns every piece of scan-time state — a dense transition table,
//! the per-state output head, and the output chain — and is built directly
//! from the NFA (trie + failure links) produced by [`NfaBuilder`]. The
//! double-array encoding is never constructed, so there is no borrowed
//! automaton and no lifetime parameter on the scanner.
//!
//! Matching performs exactly one array lookup per byte:
//! `state = dense[state * 256 + byte]`.
//!
//! Memory: `num_states × 256 × 4` bytes for the transition table (e.g. ~5 MB
//! for 5000 states) plus the output tables.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::num::NonZeroU32;

use crate::errors::{DaachorseError, Result};
use crate::nfa_builder::{NfaBuilder, DEAD_STATE_ID, ROOT_STATE_ID};
use crate::serializer::{Serializable, SerializableVec};
use crate::utils::FromU32;
use crate::{Match, MatchKind, Output};

/// A dense-DFA Aho-Corasick scanner with no double-array dependency.
pub struct ClamavFastScanner<V> {
    /// Flat dense transition table: `dense[state * 256 + byte]` = next state id.
    dense: Box<[u32]>,
    /// Output head position for each state (matches [`Output`] chain index).
    state_outputs: Box<[Option<NonZeroU32>]>,
    /// Flattened output chain (value + length + parent).
    outputs: Box<[Output<V>]>,
}

impl<V> ClamavFastScanner<V>
where
    V: Copy,
{
    /// Builds a self-contained scanner from `(pattern, value)` pairs using the
    /// standard match semantics.
    ///
    /// # Errors
    ///
    /// Returns an error when the pattern set exceeds implementation limits
    /// (see [`NfaBuilder::add`]).
    pub fn with_values<I>(patterns: I) -> Result<Self>
    where
        I: IntoIterator<Item = (Vec<u8>, V)>,
    {
        let mut nfa = NfaBuilder::<u8, V>::new(MatchKind::Standard);
        for (pattern, value) in patterns {
            nfa.add(&pattern, value)?;
        }
        let q = nfa.build_fails();
        nfa.build_outputs(&q);

        let num_states = nfa.states.len();
        let mut dense = alloc::vec![ROOT_STATE_ID; num_states * 256];
        let mut state_outputs = Vec::with_capacity(num_states);
        for s in 0..num_states {
            state_outputs.push(nfa.states[s].output_pos.get());
            let base = s * 256;
            for c in 0..=255u8 {
                dense[base + usize::from(c)] = dfa_next(&nfa, s as u32, c);
            }
        }

        Ok(Self {
            dense: dense.into_boxed_slice(),
            state_outputs: state_outputs.into_boxed_slice(),
            outputs: nfa.outputs.into_boxed_slice(),
        })
    }

    /// Returns an iterator of overlapping matches in the given haystack.
    ///
    /// Each byte consumed performs exactly **one** array lookup:
    /// `state = dense[state * 256 + byte]`.
    pub fn find_iter<P>(&self, haystack: P) -> ClamavFindIterator<'_, P, V>
    where
        P: AsRef<[u8]>,
    {
        ClamavFindIterator {
            dense: &self.dense,
            state_outputs: &self.state_outputs,
            outputs: &self.outputs,
            haystack,
            state_id: ROOT_STATE_ID,
            pos: 0,
            output_pos: self.state_outputs[usize::from_u32(ROOT_STATE_ID)],
        }
    }

    /// Returns the dense transition table (ClamAV-style).
    pub fn dense_table(&self) -> &[u32] {
        &self.dense
    }

    /// Serializes the scanner into a [`Vec`].
    #[must_use]
    pub fn serialize(&self) -> Vec<u8>
    where
        V: Serializable,
    {
        let mut result = Vec::new();
        serialize_slice(&self.dense, &mut result);
        serialize_slice(&self.state_outputs, &mut result);
        serialize_slice(&self.outputs, &mut result);
        result
    }

    /// Deserializes the scanner from the given slice.
    ///
    /// # Errors
    ///
    /// [`DaachorseError`] is returned when the given data is invalid.
    pub fn deserialize(source: &[u8]) -> Result<(Self, &[u8])>
    where
        V: Serializable,
    {
        let (dense, source) = Vec::<u32>::deserialize_from_slice(source)?;
        let (state_outputs, source) = Vec::<Option<NonZeroU32>>::deserialize_from_slice(source)?;
        let (outputs, source) = Vec::<Output<V>>::deserialize_from_slice(source)?;
        if dense.len() % 256 != 0 || state_outputs.len() * 256 != dense.len() {
            return Err(DaachorseError::invalid_automaton());
        }
        Ok((
            Self {
                dense: dense.into_boxed_slice(),
                state_outputs: state_outputs.into_boxed_slice(),
                outputs: outputs.into_boxed_slice(),
            },
            source,
        ))
    }
}

/// Serialize a slice of [`Serializable`] elements (len + items).
#[inline(always)]
fn serialize_slice<S: Serializable>(slice: &[S], dst: &mut Vec<u8>) {
    u32::try_from(slice.len()).unwrap().serialize_to_vec(dst);
    slice.iter().for_each(|x| x.serialize_to_vec(dst));
}

/// DFA closure over the trie: follow the explicit edge, otherwise the failure
/// link (ClamAV's `cli_dfa_step` behaviour), resetting at the root.
#[inline(always)]
fn dfa_next<V>(nfa: &NfaBuilder<u8, V>, mut state: u32, c: u8) -> u32
where
    V: Copy,
{
    loop {
        if let Some(&child) = nfa.states[usize::from_u32(state)].edges.get(&c) {
            return child;
        }
        if state == ROOT_STATE_ID {
            return ROOT_STATE_ID;
        }
        let fail = nfa.states[usize::from_u32(state)].fail.get();
        if fail == DEAD_STATE_ID {
            return ROOT_STATE_ID;
        }
        state = fail;
    }
}

/// Iterator created by [`ClamavFastScanner::find_iter`].
pub struct ClamavFindIterator<'a, P, V> {
    dense: &'a [u32],
    state_outputs: &'a [Option<NonZeroU32>],
    outputs: &'a [Output<V>],
    haystack: P,
    state_id: u32,
    pos: usize,
    output_pos: Option<NonZeroU32>,
}

impl<P, V> Iterator for ClamavFindIterator<'_, P, V>
where
    P: AsRef<[u8]>,
    V: Copy,
{
    type Item = Match<V>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // Yield multiple matches ending at the current position (overlapping).
        if let Some(output_pos) = self.output_pos {
            let out = &self.outputs[usize::from_u32(output_pos.get() - 1)];
            self.output_pos = out.parent();
            return Some(Match {
                length: usize::from_u32(out.length()),
                end: self.pos,
                value: out.value(),
            });
        }
        let haystack = self.haystack.as_ref();
        let len = haystack.len();
        while self.pos < len {
            let c = haystack[self.pos];
            // ── ONE array lookup per byte (ClamAV-style dense table) ─────
            self.state_id = unsafe {
                *self
                    .dense
                    .get_unchecked(self.state_id as usize * 256 + c as usize)
            };
            self.pos += 1;
            if let Some(output_pos) = unsafe {
                self.state_outputs
                    .get_unchecked(self.state_id as usize)
            } {
                let out = unsafe {
                    self.outputs
                        .get_unchecked(usize::from_u32(output_pos.get() - 1))
                };
                self.output_pos = out.parent();
                return Some(Match {
                    length: usize::from_u32(out.length()),
                    end: self.pos,
                    value: out.value(),
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_matches() {
        let sc = ClamavFastScanner::with_values(
            [b"bcd".to_vec(), b"ab".to_vec(), b"a".to_vec()]
                .into_iter()
                .enumerate()
                .map(|(i, p)| (p, i as u32)),
        )
        .unwrap();
        let ms: Vec<(usize, usize, u32)> = sc
            .find_iter("abcd")
            .map(|m| (m.start(), m.end(), m.value()))
            .collect();
        assert_eq!(ms, vec![(0, 1, 2), (0, 2, 1), (1, 4, 0)]);
    }

    #[test]
    fn no_match_returns_none() {
        let sc = ClamavFastScanner::with_values(
            [b"xyz".to_vec()]
                .into_iter()
                .enumerate()
                .map(|(i, p)| (p, i as u32)),
        )
        .unwrap();
        assert!(sc.find_iter("---abc---").next().is_none());
    }

    #[test]
    fn empty_pattern_set() {
        let sc = ClamavFastScanner::with_values(Vec::<(Vec<u8>, u32)>::new()).unwrap();
        assert!(sc.find_iter("anything").next().is_none());
    }

    #[test]
    fn serialize_roundtrip() {
        let patterns: Vec<(Vec<u8>, u32)> = vec![
            (b"ab".to_vec(), 0),
            (b"abc".to_vec(), 1),
            (b"xox-abc".to_vec(), 2),
            (b"AKIA1234567890ABC".to_vec(), 3),
        ];
        let sc = ClamavFastScanner::with_values(patterns).unwrap();
        let bytes = sc.serialize();
        let (sc2, rest) = ClamavFastScanner::<u32>::deserialize(&bytes).unwrap();
        assert!(rest.is_empty());
        let hay = b"ab abcd xox-abc AKIA1234567890ABC nothing";
        let a: Vec<(usize, usize, u32)> = sc
            .find_iter(hay)
            .map(|m| (m.start(), m.end(), m.value()))
            .collect();
        let b: Vec<(usize, usize, u32)> = sc2
            .find_iter(hay)
            .map(|m| (m.start(), m.end(), m.value()))
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn differential_vs_double_array() {
        use crate::DoubleArrayAhoCorasick;

        let patterns: Vec<Vec<u8>> = vec![
            b"ab".to_vec(),
            b"abc".to_vec(),
            b"bcd".to_vec(),
            b"cd".to_vec(),
            b"a".to_vec(),
            b"zzzz".to_vec(),
            b"key=value".to_vec(),
            b"xox-abc-123".to_vec(),
            b"AKIA0123456789ABC".to_vec(),
        ];
        let haystacks: Vec<&[u8]> = vec![
            b"abcd",
            b"key=value and xox-abc-123",
            b"AKIA0123456789ABCDEF",
            b"no matches here",
            b"ab abcd a bcd cd",
            b"",
            b"zz",
        ];

        let fast = ClamavFastScanner::with_values(
            patterns
                .iter()
                .enumerate()
                .map(|(i, p)| (p.clone(), i as u32)),
        )
        .unwrap();
        let pma = DoubleArrayAhoCorasick::with_values(
            patterns
                .iter()
                .enumerate()
                .map(|(i, p)| (p.as_slice(), i as u32)),
        )
        .unwrap();

        for hay in &haystacks {
            let a: Vec<(usize, usize, u32)> = fast
                .find_iter(hay)
                .map(|m| (m.start(), m.end(), m.value()))
                .collect();
            let b: Vec<(usize, usize, u32)> = pma
                .find_overlapping_iter(hay)
                .map(|m| (m.start(), m.end(), m.value()))
                .collect();
            assert_eq!(a, b, "haystack: {:?}", hay);
        }
    }
}
