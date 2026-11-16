//! The guest's witness for one child proof, and the opening shapes it shares with the guest's constants.

use super::cells::count;
use super::error::AggregateError;
use super::hints::SubHints;
use fiat_shamir_prover::merkle::{hash_pair, hash_to_scalars, hash_words};
use lean_vm_core::cpu::{
    KAPPA_SOURCE, QKECCAK, block_kappa_sources, col_kappa_sources, from_scalar, layout_of, piece_digits,
};
use lean_vm_core::keccak::{SLOT_LOG, TABLE_BITS, TABLES, slot_blocks};
use lean_vm_core::leaf::{self, Block};
use primitives::field::F128;
use verifier::layout::N_TABLES;

/// The stacked WHIR opening of a `2^mu` stack at a rate: the verifier's config, then each level's message columns
/// after its folds and its Merkle tree's depth. The hint builder needs it for the real opening and the placeholder map
/// for every candidate size, so it lives here: a candidate whose shape differed from the real one would compile a
/// guest that cannot open the proof it is handed. `None` for a shape no proof has.
pub(super) fn whir_levels(mu: usize, log_inv_rate: usize) -> Option<(verifier::WhirConfig, Vec<usize>, Vec<usize>)> {
    let config = verifier::derive_config(mu, log_inv_rate).ok()?;
    let log_msg_cols: Vec<usize> = config
        .folds
        .iter()
        .scan(mu, |cols, &fold| {
            *cols -= fold;
            Some(*cols)
        })
        .collect();
    let depth = log_msg_cols
        .iter()
        .zip(&config.log_inv_rates)
        .map(|(c, r)| c + r)
        .collect();
    Some((config, log_msg_cols, depth))
}

/// The path bits a query's Merkle walk dispatches on at a time, the guest's `QW` (`walk_path`).
pub(super) const QUERY_WINDOW_BITS: usize = 6;
/// The cells `decode_query_windows` holds a squeezed word's windows in, the guest's `QUERY_WINDOWS_CAP`.
pub(super) const QUERY_WINDOWS_CAP: usize = 36;

/// The depth of the cap a level's paths are walked into, which leaves the walked part whole windows of
/// [`QUERY_WINDOW_BITS`] bits: at least two levels deep, and from `log2(queries) - 5` to five levels below that.
pub(super) fn merkle_cap_depth(queries: usize, depth: usize) -> usize {
    let near = (queries.next_power_of_two().ilog2() as usize)
        .saturating_sub(QUERY_WINDOW_BITS - 1)
        .max(2)
        .min(depth);
    depth - (depth - near) / QUERY_WINDOW_BITS * QUERY_WINDOW_BITS
}

/// The first column of each column group: a maximal run of committed columns, in compact order, sharing one kappa
/// source. The stacked order (descending kappa, then index) keeps a group's columns consecutive, since any other
/// column of equal kappa lies outside it.
pub(super) fn col_groups(sources: &[(usize, usize)]) -> Vec<usize> {
    (0..sources.len())
        .filter(|&c| c == 0 || sources[c] != sources[c - 1])
        .collect()
}

/// Whether a truncated proof cuts a column of this kappa source: the memory's and the tables' are, and the
/// slot blocks of the halves of `q_keccak` (see [`guest_col_sources`]).
pub(super) fn cut(source: (usize, usize)) -> bool {
    source.0 >= 1 && (source.1 == 0 || is_half(source))
}

/// The slot blocks of the halves of `q_keccak`, each a column of the guest: `(half, first slot, log slots)`, by half
/// then block. `KECCAK` is one block of all 64 slots, `KECCAK64` three that leave out its public slots.
pub(super) fn qt_columns() -> Vec<(usize, usize, usize)> {
    (0..TABLES)
        .flat_map(|half| slot_blocks(half).iter().map(move |&(first, log)| (half, first, log)))
        .collect()
}

/// Whether a kappa source is a slot block of a half of `q_keccak`: the source of its half's rows, and the log of
/// the block's slots as the adjustment, below the whole column's `SLOT_LOG + TABLE_BITS`.
pub(super) fn is_half(source: (usize, usize)) -> bool {
    source.0 >= KAPPA_SOURCE && source.0 < KAPPA_SOURCE + TABLES && source.1 < SLOT_LOG + TABLE_BITS
}

/// The committed columns' kappa sources, then those of the slot blocks of the halves of `q_keccak`, which are
/// columns of the guest alone: their pieces are what a proof commits in place of the column, each block cut at its
/// table's rows like a table column. The guest places them with the others, in this order. The column `q_keccak`
/// itself has no words.
pub(super) fn guest_col_sources(kbc: usize) -> Vec<(usize, usize)> {
    let mut sources: Vec<(usize, usize)> = col_kappa_sources(kbc).into_iter().flatten().collect();
    sources.extend(
        qt_columns()
            .into_iter()
            .map(|(half, _, log)| (KAPPA_SOURCE + half, log)),
    );
    sources
}

/// A cut column group's slots in the stacked order: piece `j` is `2^(kappa - j)` rows, `j = 0` the whole column.
pub(super) const PIECES: usize = 5;

/// The placement slots `(group, piece)`: [`PIECES`] for a cut group, one for any other.
pub(super) fn col_slots(sources: &[(usize, usize)], groups: &[usize]) -> Vec<(usize, usize)> {
    groups
        .iter()
        .enumerate()
        .flat_map(|(g, &start)| (0..if cut(sources[start]) { PIECES } else { 1 }).map(move |j| (g, j)))
        .collect()
}

/// The first block of each block group: as [`col_groups`], over the bus blocks, a group never crossing the start of a
/// side (`side_starts`), since a side's blocks tile its own cube.
pub(super) fn block_groups(sources: &[(usize, usize)], side_starts: &[usize]) -> Vec<usize> {
    (0..sources.len())
        .filter(|&b| side_starts.contains(&b) || sources[b] != sources[b - 1])
        .collect()
}

/// `count(g)` for each run of one group `g` in `groups`, in order. A group that came back after another would be
/// named twice, which the caller's count of runs catches.
fn group_runs(groups: impl Iterator<Item = usize>) -> Vec<F128> {
    let mut runs: Vec<usize> = groups.collect();
    runs.dedup();
    runs.into_iter().map(count).collect()
}

/// The guest's witness for verifying one child (`guests/lean_ethereum.py`), built from the layout of an execution of
/// the `2^kbc`-instruction inner program at the sizes the `summary` the `verifier` crate read out of the child's
/// `proof` announces, and from that proof's Merkle paths, whose positions and sizes its replay pinned. Nothing here
/// says the child verifies: the guest's run does.
pub(super) fn gen_verify(
    kbc: usize,
    proof: &verifier::Proof,
    summary: &verifier::Summary,
) -> Result<SubHints, AggregateError> {
    let rows = std::array::from_fn(|t| summary.table_rows[t] as usize);
    let layout = layout_of(kbc, summary.log_memory, rows);
    let taus = layout.taus;
    let sides = [layout.push, layout.pull, layout.count];
    // The guest's buffers have fixed capacities, so its constants do not depend on a child's shape. A child that
    // verifies fits them.
    let fits = sides.iter().all(|blocks| leaf::layout(blocks).mu <= MU_CAP) && proof.stream.len() <= STREAM_CAP;
    if !fits {
        return Err(AggregateError::InvalidChild);
    }
    let mut stream: Vec<F128> = proof.stream.iter().copied().map(from_scalar).collect();
    stream.resize(STREAM_CAP, F128::ZERO);

    // The guest replays the proof off the same stream the native verifier reads: every transmitted scalar is already
    // there in protocol order.
    let mut hints: SubHints = vec![
        ("stream", stream),
        // The table sumcheck's round count, which the guest certifies is the largest of the taus.
        ("zc_tau_max", vec![count(*taus.iter().max().unwrap())]),
        // Every table's log height, which the guest certifies against its row count.
        ("table_logs", taus.iter().map(|&t| count(t)).collect()),
        // Which of the two hash tables is the taller, the second's pick: the height the reduction completes both to.
        (
            "kappa_pick",
            vec![F128::from(u128::from(
                taus[verifier::layout::OP_KECCAK64] > taus[verifier::layout::OP_KECCAK],
            ))],
        ),
        ("col_slot_order", col_slot_order(kbc, summary, &taus)),
        ("block_group_order", block_group_order(kbc, &sides)),
    ];
    let (config, _, depth) =
        whir_levels(summary.stack_log, summary.log_inv_rate).ok_or(AggregateError::InvalidChild)?;
    hints.extend(merkle_hints(config.queries, &depth, proof));
    Ok(hints)
}

/// The placement slots ([`col_slots`]) in their stacking order: the slots a proof of this `summary` commits first, by
/// descending kappa then first column, the absent ones after them in slot order. A truncated proof has no layout of its
/// own to read the order off, so it follows from the sizes the proof announces; the guest certifies it before it reads
/// claims by it.
fn col_slot_order(kbc: usize, summary: &verifier::Summary, taus: &[usize; N_TABLES]) -> Vec<F128> {
    let sources = col_kappa_sources(kbc);
    // Compact index to global column.
    let committed: Vec<usize> = (0..sources.len()).filter(|&c| sources[c].is_some()).collect();
    let cks = guest_col_sources(kbc);
    let groups = col_groups(&cks);
    let truncation = summary.truncation;
    let source_log = |source: usize| match source {
        0 => 0,
        1 => summary.log_memory,
        // The hash tables are completed to the taller one's height, and so are the halves of `q_keccak`.
        s if s >= KAPPA_SOURCE => verifier::layout::HASH_TABLES
            .iter()
            .map(|&t| taus[t])
            .max()
            .expect("hash tables"),
        s => taus[s - 2],
    };
    let digits = |source: usize| {
        let k = source_log(source);
        let real = match truncation {
            // The blocks of `q_keccak` are cut at their table's rows in every proof.
            _ if source >= KAPPA_SOURCE => {
                summary.table_rows[verifier::layout::HASH_TABLES[source - KAPPA_SOURCE]] as usize
            }
            None => 1 << k,
            Some(t) if source == 1 => t.mem_count as usize,
            Some(_) => summary.table_rows[source - 2] as usize,
        };
        piece_digits(k, real)
    };
    let qkeccak = committed.iter().position(|&global| global == QKECCAK);
    let mut slots: Vec<(bool, usize, usize, usize)> = col_slots(&cks, &groups)
        .into_iter()
        .enumerate()
        .map(|(slot, (g, j))| {
            let (source, adj) = cks[groups[g]];
            // The pieces of `q_keccak` replace the column: it has no words.
            let present = if cut((source, adj)) {
                digits(source)[j]
            } else {
                Some(groups[g]) != qkeccak
            };
            (present, source_log(source) + adj - j, groups[g], slot)
        })
        .collect();
    slots.sort_by_key(|&(present, kappa, start, slot)| match present {
        true => (0, std::cmp::Reverse(kappa), start),
        false => (1, std::cmp::Reverse(0), slot),
    });
    slots.into_iter().map(|(.., slot)| count(slot)).collect()
}

/// Per bus side, its blocks in their stacking order (`leaf::layout`'s offsets), each run of a block group
/// ([`block_groups`]) named once. The guest certifies the native verifier's order (descending kappa, then index)
/// before it derives offsets from it.
fn block_group_order(kbc: usize, sides: &[Vec<Block>; 3]) -> Vec<F128> {
    let mut side_starts = vec![0];
    for blocks in sides {
        side_starts.push(side_starts.last().unwrap() + blocks.len());
    }
    let groups = block_groups(&block_kappa_sources(kbc), &side_starts);
    let mut stacked = Vec::new();
    for (blocks, start) in sides.iter().zip(&side_starts) {
        let offsets = leaf::layout(blocks).offsets;
        let mut order: Vec<usize> = (0..blocks.len()).collect();
        order.sort_by_key(|&i| offsets[i]);
        stacked.extend(order.into_iter().map(|i| start + i));
    }
    // A group never crosses a side, so runs never merge across one.
    let order = group_runs(
        stacked
            .into_iter()
            .map(|block| groups.partition_point(|&start| start <= block) - 1),
    );
    assert_eq!(order.len(), groups.len(), "a block group's blocks are consecutive");
    order
}

/// The stacked WHIR opening's Merkle witness, per level: each query's leaf row and the children of every node it
/// climbs through below the level's cap, then the cap, every node of the top [`merkle_cap_depth`] levels the queries
/// reach, and which of its nodes they reach. The upper tree is shared by a level's queries; a subtree no query reaches
/// stays zero, unread.
fn merkle_hints(queries: &[usize], depth: &[usize], proof: &verifier::Proof) -> SubHints {
    let mut hints = Vec::new();
    let levels = queries.iter().zip(depth);
    for (level, ((&queries, &depth), phase)) in levels.zip(&proof.merkle_openings).enumerate() {
        let cap_depth = merkle_cap_depth(queries, depth);
        let path_depth = depth - cap_depth;
        let mut nodes = vec![[0u8; 32]; 2 << cap_depth];
        let mut active = vec![F128::ZERO; 1 << cap_depth];
        for opening in phase {
            let leaf: Vec<u64> = opening.leaf.iter().map(|word| word.0).collect();
            // Level 0's rows are K lanes, a cell each; deeper ones E words, two lanes a cell.
            let row = if level == 0 {
                leaf.iter().map(|&x| F128::new(x, 0)).collect()
            } else {
                leaf.as_chunks().0.iter().map(|&[c0, c1]| F128::new(c0, c1)).collect()
            };
            hints.push(("merkle_leaf_rows", row));
            let mut path_children = Vec::with_capacity(4 * path_depth);
            let mut node = hash_words(leaf);
            let mut index = (1 << depth) + opening.leaf_index as usize;
            for (height, sibling) in opening.siblings.iter().enumerate() {
                let (left, right) = if index & 1 == 0 {
                    (&node, sibling)
                } else {
                    (sibling, &node)
                };
                if height < path_depth {
                    path_children.extend(hash_to_scalars::<F128>(left));
                    path_children.extend(hash_to_scalars::<F128>(right));
                } else {
                    nodes[index] = node;
                    nodes[index ^ 1] = *sibling;
                    active[index >> 1] = F128::ONE;
                }
                node = hash_pair(left, right);
                index >>= 1;
            }
            nodes[1] = node;
            hints.push(("merkle_children", path_children));
        }
        hints.push(("merkle_caps", nodes.iter().flat_map(hash_to_scalars).collect()));
        hints.push(("merkle_cap_active", active));
    }
    hints
}

/// The guest's baked buffer caps, which `placeholder_map` compiles in and
/// `gen_verify` admits against: one definition, so a hinted shape can never
/// outgrow the buffer the guest was compiled with.
pub(super) const MU_CAP: usize = 40;
pub(super) const STREAM_CAP: usize = 8192;
