"""Enumerate atom relations for classical bottom-permutation-top bases.

Run with Sage's Python, for example:

    sage -python classical_atom_relations.py A3 --family all --count-only
    sage -python classical_atom_relations.py C3 --family OR4 --limit 5
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import permutations, product

sys.dont_write_bytecode = True

from sage.all import QQ, matrix, vector

from classical_bpt import (
    BranchingRule,
    ClassicalBPT,
    Order,
    Tensor,
    fmt_order,
    parse_cartan_type,
)


@dataclass(frozen=True)
class Atom:
    # A non-identity branching rule, treated as a generating atom.
    context: ClassicalBPT
    rule: BranchingRule

    @property
    def domain(self) -> Order:
        return self.rule.domain_order

    @property
    def codomain(self) -> Order:
        return self.rule.codomain_order


@dataclass(frozen=True)
class EmbeddedAtom:
    # An atom with identity strands placed on its left and right.
    atom: Atom
    left: Order
    right: Order

    @property
    def context(self) -> ClassicalBPT:
        return self.atom.context

    @property
    def domain(self) -> Order:
        return self.left + self.atom.domain + self.right

    @property
    def codomain(self) -> Order:
        return self.left + self.atom.codomain + self.right

    def word(self, flipped: bool = False) -> str:
        return self.context.tensor_embed(self.context.branching_symbol(self.atom.rule, flipped=flipped), self.left, self.right)

    def apply(self, tensor: Tensor, inverse: bool = False) -> Tensor | None:
        block_order = self.codomain if inverse else self.domain
        block_width = len(self.atom.codomain if inverse else self.atom.domain)
        pos = len(self.left)
        if len(tensor) != len(block_order):
            return None
        image = self.context.apply_branching_rule(self.atom.rule, tuple(tensor[pos : pos + block_width]), inverse=inverse)
        if image is None:
            return None
        return tensor[:pos] + image + tensor[pos + block_width :]


@dataclass(frozen=True)
class AtomComposite:
    # HI1: a flipped atom followed by an atom on a common middle word.
    middle: Order
    top: EmbeddedAtom
    bottom: EmbeddedAtom

    @property
    def family(self) -> str:
        return "HI1"

    @property
    def context(self) -> ClassicalBPT:
        return self.top.context

    @property
    def domain(self) -> Order:
        return self.top.codomain

    @property
    def codomain(self) -> Order:
        return self.bottom.codomain

    def word(self) -> str:
        return f"{self.bottom.word()} o {self.top.word(flipped=True)}"

    def apply(self, tensor: Tensor) -> Tensor | None:
        middle_tensor = self.top.apply(tensor, inverse=True)
        if middle_tensor is None:
            return None
        return self.bottom.apply(middle_tensor)


@dataclass(frozen=True)
class CrossingComposite:
    # HI2: the same atom-over-atom picture, with a middle permutation inserted.
    top: EmbeddedAtom
    bottom: EmbeddedAtom
    positions: tuple[int, ...] = (0,)

    @property
    def family(self) -> str:
        return "HI2"

    @property
    def context(self) -> ClassicalBPT:
        return self.top.context

    @property
    def domain(self) -> Order:
        return self.top.codomain

    @property
    def codomain(self) -> Order:
        return self.bottom.codomain

    def crossing_terms(self) -> tuple[str, ...]:
        order = self.top.domain
        terms: list[str] = []
        for pos in self.positions:
            terms.append(self.context.tensor_embed(self.context.crossing_symbol(order[pos : pos + 2]), order[:pos], order[pos + 2 :]))
            order = swap_order(order, pos)
        return tuple(terms)

    def word(self) -> str:
        return self.context.compose_terms([self.top.word(flipped=True), *self.crossing_terms(), self.bottom.word()])

    def apply(self, tensor: Tensor) -> Tensor | None:
        outputs = self.top.apply(tensor, inverse=True)
        if outputs is None:
            return None
        order = self.top.domain
        for pos in self.positions:
            crossed = self.context.sage_commutor(order[pos : pos + 2], outputs[pos : pos + 2])
            outputs = outputs[:pos] + crossed + outputs[pos + 2 :]
            order = swap_order(order, pos)
        return self.bottom.apply(outputs)


@dataclass(frozen=True)
class BottomCrossingComposite:
    # HI3: a permutation is precomposed with a single atom.
    bottom: EmbeddedAtom
    positions: tuple[int, ...]

    @property
    def family(self) -> str:
        return "HI3"

    @property
    def context(self) -> ClassicalBPT:
        return self.bottom.context

    @property
    def domain(self) -> Order:
        order = self.bottom.domain
        for pos in reversed(self.positions):
            order = order[:pos] + (order[pos + 1], order[pos]) + order[pos + 2 :]
        return order

    @property
    def codomain(self) -> Order:
        return self.bottom.codomain

    def crossing_terms(self) -> tuple[str, ...]:
        order = self.domain
        terms: list[str] = []
        for pos in self.positions:
            terms.append(
                self.context.tensor_embed(
                    self.context.crossing_symbol(order[pos : pos + 2]),
                    order[:pos],
                    order[pos + 2 :],
                )
            )
            order = order[:pos] + (order[pos + 1], order[pos]) + order[pos + 2 :]
        return tuple(terms)

    def word(self) -> str:
        return self.context.compose_terms(list(self.crossing_terms()) + [self.bottom.word()])

    def apply(self, tensor: Tensor) -> Tensor | None:
        outputs = tuple(tensor)
        order = self.domain
        for pos in self.positions:
            crossed = self.context.sage_commutor(order[pos : pos + 2], outputs[pos : pos + 2])
            outputs = outputs[:pos] + crossed + outputs[pos + 2 :]
            order = order[:pos] + (order[pos + 1], order[pos]) + order[pos + 2 :]
        return self.bottom.apply(outputs)


@dataclass(frozen=True)
class RightCrossingThenAtomComposite:
    # OR1: the crossing immediately before an atom at the right end.
    atom: EmbeddedAtom
    right_label: int

    @property
    def family(self) -> str:
        return "OR1"

    @property
    def context(self) -> ClassicalBPT:
        return self.atom.context

    @property
    def domain(self) -> Order:
        order = self.atom.domain
        return order[:-2] + (order[-1], order[-2])

    @property
    def codomain(self) -> Order:
        return self.atom.codomain

    def crossing_word(self) -> str:
        return self.context.tensor_embed(self.context.crossing_symbol(self.domain[-2:]), self.domain[:-2], ())

    def word(self) -> str:
        return self.context.compose_terms([self.crossing_word(), self.atom.word()])

    def apply(self, tensor: Tensor) -> Tensor | None:
        crossed = self.context.sage_commutor(self.domain[-2:], tensor[-2:])
        after_crossing = tensor[:-2] + crossed
        return self.atom.apply(after_crossing)


@dataclass(frozen=True)
class OverlappingAtomComposite:
    # OR2: two atoms overlap, possibly with a middle permutation.
    middle: Order
    atom_a: EmbeddedAtom
    atom_b: EmbeddedAtom
    positions: tuple[int, ...] = ()

    @property
    def family(self) -> str:
        return "OR2"

    @property
    def context(self) -> ClassicalBPT:
        return self.atom_a.context

    @property
    def domain(self) -> Order:
        return self.atom_b.domain

    @property
    def codomain(self) -> Order:
        return self.atom_a.codomain

    def crossing_terms(self) -> tuple[str, ...]:
        order = self.atom_b.codomain
        terms: list[str] = []
        for pos in self.positions:
            terms.append(self.context.tensor_embed(self.context.crossing_symbol(order[pos : pos + 2]), order[:pos], order[pos + 2 :]))
            order = swap_order(order, pos)
        return tuple(terms)

    def word(self) -> str:
        return self.context.compose_terms([self.atom_b.word(), *self.crossing_terms(), self.atom_a.word()])

    def apply(self, tensor: Tensor) -> Tensor | None:
        outputs = self.atom_b.apply(tensor)
        if outputs is None:
            return None
        order = self.atom_b.codomain
        for pos in self.positions:
            crossed = self.context.sage_commutor(order[pos : pos + 2], outputs[pos : pos + 2])
            outputs = outputs[:pos] + crossed + outputs[pos + 2 :]
            order = swap_order(order, pos)
        return self.atom_a.apply(outputs)

@dataclass(frozen=True)
class RotateRightStrandThenAtomComposite:
    # OR3: the rightmost strand passes leftward across the domain of an atom.
    atom: EmbeddedAtom
    crossing_terms: tuple[str, ...]

    @property
    def family(self) -> str:
        return "OR3"

    @property
    def context(self) -> ClassicalBPT:
        return self.atom.context

    @property
    def domain(self) -> Order:
        left_label = self.atom.left[0]
        return self.atom.atom.domain + (left_label,)

    @property
    def codomain(self) -> Order:
        return self.atom.codomain

    def word(self) -> str:
        return self.context.compose_terms(list(self.crossing_terms) + [self.atom.word()])

    def apply(self, tensor: Tensor) -> Tensor | None:
        outputs = tuple(tensor)
        order = self.domain
        for pos in range(len(self.atom.atom.domain) - 1, -1, -1):
            crossed = self.context.sage_commutor(order[pos : pos + 2], outputs[pos : pos + 2])
            outputs = outputs[:pos] + crossed + outputs[pos + 2 :]
            order = order[:pos] + (order[pos + 1], order[pos]) + order[pos + 2 :]
        return self.atom.apply(outputs)


@dataclass(frozen=True)
class ReidemeisterThreeComposite:
    # OR4: the three-strand Reidemeister move for three distinct labels.
    context: ClassicalBPT
    labels: Order
    crossing_terms: tuple[str, ...]

    @property
    def family(self) -> str:
        return "OR4"

    @property
    def domain(self) -> Order:
        return self.labels

    @property
    def codomain(self) -> Order:
        return (self.labels[2], self.labels[1], self.labels[0])

    def word(self) -> str:
        return self.context.compose_terms(list(self.crossing_terms))

    def apply(self, tensor: Tensor) -> Tensor | None:
        outputs = tuple(tensor)
        order = self.domain
        for pos in (1, 0, 1):
            crossed = self.context.sage_commutor(order[pos : pos + 2], outputs[pos : pos + 2])
            outputs = outputs[:pos] + crossed + outputs[pos + 2 :]
            order = order[:pos] + (order[pos + 1], order[pos]) + order[pos + 2 :]
        return outputs


def atoms(context: ClassicalBPT, max_atom_len: int | None = None) -> tuple[Atom, ...]:
    # Definition 2B.8 gives one atom for each element of a fundamental crystal.
    atom_list = [
        Atom(context, rule)
        for rule in context.all_atom_rules()
        if rule.old_suffix and (max_atom_len is None or len(rule.domain_order) <= max_atom_len)
    ]
    return tuple(sorted(atom_list, key=lambda atom: (len(atom.domain), atom.domain, atom.codomain, atom.rule.name)))


def common_middle(top: Atom, bottom: Atom, offset: int) -> tuple[Order, EmbeddedAtom, EmbeddedAtom] | None:
    # Put two atoms in a minimal common word, allowing identities on either side.
    positions: dict[int, int] = {}
    for idx, label in enumerate(top.domain):
        positions[idx] = label
    for idx, label in enumerate(bottom.domain):
        pos = offset + idx
        if pos in positions and positions[pos] != label:
            return None
        positions[pos] = label

    start = min(positions)
    end = max(positions) + 1
    middle = tuple(positions[pos] for pos in range(start, end))
    top_pos = -start
    bottom_pos = offset - start
    top_embedding = EmbeddedAtom(top, middle[:top_pos], middle[top_pos + len(top.domain) :])
    bottom_embedding = EmbeddedAtom(bottom, middle[:bottom_pos], middle[bottom_pos + len(bottom.domain) :])
    return middle, top_embedding, bottom_embedding


def atom_composites(atom_list: tuple[Atom, ...]) -> tuple[AtomComposite, ...]:
    out: list[AtomComposite] = []
    seen = set()
    for top in atom_list:
        for bottom in atom_list:
            for offset in range(1 - len(bottom.domain), len(top.domain)):
                found = common_middle(top, bottom, offset)
                if found is None:
                    continue
                middle, top_embedding, bottom_embedding = found
                key = (middle, top_embedding, bottom_embedding)
                if key in seen:
                    continue
                seen.add(key)
                out.append(AtomComposite(middle, top_embedding, bottom_embedding))
    return tuple(out)


def common_prefix(first: Order, second: Order) -> Order | None:
    # Merge two words that start at the same position; the longer word supplies padding.
    width = min(len(first), len(second))
    if first[:width] != second[:width]:
        return None
    return first if len(first) >= len(second) else second


def crossing_composites(atom_list: tuple[Atom, ...]) -> tuple[CrossingComposite, ...]:
    # Enumerate HI2 relations with the deterministic middle permutation between the atoms.
    return _all_crossing_composites(atom_list)


def single_crossing_composites(atom_list: tuple[Atom, ...]) -> tuple[CrossingComposite, ...]:
    # The original HI2 cases: one crossing between the left identity strand and the atom.
    out: list[CrossingComposite] = []
    seen = set()
    for top in atom_list:
        for left_label in top.context.index_set:
            if left_label == top.domain[0]:
                continue
            after_crossing_tail = (left_label,) + top.domain[1:]
            for bottom in atom_list:
                shared_tail = common_prefix(after_crossing_tail, bottom.domain)
                if shared_tail is None:
                    continue
                top_right = shared_tail[len(after_crossing_tail) :]
                bottom_right = shared_tail[len(bottom.domain) :]
                top_embedding = EmbeddedAtom(top, (left_label,), top_right)
                bottom_embedding = EmbeddedAtom(bottom, (top.domain[0],), bottom_right)
                composite = CrossingComposite(top_embedding, bottom_embedding)
                key = (top_embedding, bottom_embedding, composite.positions)
                if key in seen:
                    continue
                seen.add(key)
                out.append(composite)
    return tuple(out)


def swap_order(order: Order, pos: int) -> Order:
    return order[:pos] + (order[pos + 1], order[pos]) + order[pos + 2 :]


def left_reidemeister_swaps(start: tuple[int, ...]) -> tuple[int, ...]:
    labels = list(start)
    swaps: list[int] = []
    for desired in range(len(labels)):
        idx = labels.index(desired)
        for pos in range(idx - 1, desired - 1, -1):
            labels[pos], labels[pos + 1] = labels[pos + 1], labels[pos]
            swaps.append(pos)
    return tuple(swaps)


def multiset_contains(word: Order, subword: Order) -> bool:
    return not (Counter(subword) - Counter(word))


def distinct_permutations(order: Order) -> set[Order]:
    return set(permutations(order))


def stable_target_permutation(start: Order, target: Order) -> tuple[int, ...]:
    # Equal-labelled strands are matched in order, since they are not allowed to cross.
    buckets: dict[int, list[int]] = {}
    for idx, label in enumerate(start):
        buckets.setdefault(label, []).append(idx)
    used = {label: 0 for label in buckets}
    out: list[int] = []
    for label in target:
        out.append(buckets[label][used[label]])
        used[label] += 1
    return tuple(out)


def crossing_positions_from_target(start: Order, target: Order) -> tuple[int, ...]:
    # Endpoints determine the left-Reidemeister crossing word.
    return tuple(reversed(left_reidemeister_swaps(stable_target_permutation(start, target))))


def admissible_crossing_positions(start: Order, positions: tuple[int, ...], protect_last: bool = True) -> bool:
    labels = list(start)
    strands = list(range(len(start)))
    last = len(start) - 1
    for pos in positions:
        if labels[pos] == labels[pos + 1]:
            return False
        if protect_last and (strands[pos] == last or strands[pos + 1] == last):
            return False
        labels[pos], labels[pos + 1] = labels[pos + 1], labels[pos]
        strands[pos], strands[pos + 1] = strands[pos + 1], strands[pos]
    return True


def crossing_positions_avoid_strands(start: Order, positions: tuple[int, ...], protected: set[int]) -> bool:
    labels = list(start)
    strands = list(range(len(start)))
    for pos in positions:
        if labels[pos] == labels[pos + 1]:
            return False
        if strands[pos] in protected or strands[pos + 1] in protected:
            return False
        labels[pos], labels[pos + 1] = labels[pos + 1], labels[pos]
        strands[pos], strands[pos + 1] = strands[pos + 1], strands[pos]
    return True


def target_to_source(start: Order, target: Order) -> tuple[int, ...]:
    return stable_target_permutation(start, target)


def source_to_target(start: Order, target: Order) -> tuple[int, ...]:
    out = [0] * len(start)
    for target_pos, source_pos in enumerate(target_to_source(start, target)):
        out[source_pos] = target_pos
    return tuple(out)


def prefix_crossing_positions(start: Order, target: Order, prefix_width: int, protected: set[int]) -> tuple[int, ...] | None:
    # Return the middle permutation if it is supported in the allowed prefix.
    if prefix_width < 0 or start[prefix_width:] != target[prefix_width:]:
        return None
    positions = crossing_positions_from_target(start, target)
    if any(pos >= prefix_width - 1 for pos in positions):
        return None
    if not crossing_positions_avoid_strands(start, positions, protected):
        return None
    return positions


def no_passive_middle_strands(start: Order, target: Order, start_block: range, target_block: range) -> bool:
    # Exclude separated identity strands in the middle permutation.
    target_pos = source_to_target(start, target)
    source_pos = target_to_source(start, target)
    for idx in range(len(start)):
        if idx not in start_block and target_pos[idx] not in target_block:
            return False
    for idx in range(len(target)):
        if idx not in target_block and source_pos[idx] not in start_block:
            return False
    return True


def left_strands_interact(start: Order, target: Order, start_block: range, target_block: range) -> bool:
    # All strands to the left of the atom block must connect to the other atom.
    target_pos = source_to_target(start, target)
    source_pos = target_to_source(start, target)
    if any(target_pos[idx] not in target_block for idx in range(start_block.start)):
        return False
    if any(source_pos[idx] not in start_block for idx in range(target_block.start)):
        return False
    return True


def target_words_with_block_after_left_identity(start: Order, block: Order) -> tuple[Order, ...]:
    if len(block) + 1 > len(start) or not multiset_contains(start, block):
        return ()
    rest_counter = Counter(start) - Counter(block)
    rest: list[int] = []
    for label, count in rest_counter.items():
        rest.extend([label] * count)
    out: set[Order] = set()
    for rest_order in distinct_permutations(tuple(rest)):
        if rest_order:
            out.add(rest_order[:1] + block + rest_order[1:])
    return tuple(out)


def orders_with_block(labels: Order, width: int, block: Order, pos: int) -> tuple[Order, ...]:
    if pos < 0 or pos + len(block) > width:
        return ()
    free = width - len(block)
    out: list[Order] = []
    for filler in product(labels, repeat=free):
        out.append(filler[:pos] + block + filler[pos:])
    return tuple(out)


def _all_crossing_composites(atom_list: tuple[Atom, ...]) -> tuple[CrossingComposite, ...]:
    # Enumerate all HI2 placements allowed by the endpoint conditions.
    out: list[CrossingComposite] = []
    seen = set()
    labels = atom_list[0].context.index_set if atom_list else ()

    for top in atom_list:
        for bottom in atom_list:
            max_width = len(top.domain) + len(bottom.domain)
            for width in range(max(len(top.domain), len(bottom.domain)), max_width + 1):
                for top_pos in range(width - len(top.domain) + 1):
                    top_block = range(top_pos, top_pos + len(top.domain))
                    protected = {top_block.stop - 1}
                    for bottom_pos in range(width - len(bottom.domain) + 1):
                        bottom_block = range(bottom_pos, bottom_pos + len(bottom.domain))
                        prefix_width = min(top_block.stop - 1, bottom_block.stop - 1)
                        if prefix_width < 1:
                            continue
                        for start in orders_with_block(labels, width, top.domain, top_pos):
                            for target in orders_with_block(labels, width, bottom.domain, bottom_pos):
                                if Counter(start) != Counter(target):
                                    continue
                                target_pos = source_to_target(start, target)
                                if target_pos[top_block.stop - 1] not in bottom_block:
                                    continue
                                if not no_passive_middle_strands(start, target, top_block, bottom_block):
                                    continue
                                if not left_strands_interact(start, target, top_block, bottom_block):
                                    continue
                                positions = prefix_crossing_positions(start, target, prefix_width, protected)
                                if not positions:
                                    continue
                                top_embedding = EmbeddedAtom(top, start[:top_pos], start[top_block.stop :])
                                bottom_embedding = EmbeddedAtom(bottom, target[:bottom_pos], target[bottom_block.stop :])
                                key = (top_embedding, bottom_embedding, positions)
                                if key in seen:
                                    continue
                                seen.add(key)
                                out.append(CrossingComposite(top_embedding, bottom_embedding, positions))

    return tuple(sorted(out, key=lambda composite: (len(composite.domain), composite.domain, composite.codomain, composite.word())))


def additional_crossing_composites(atom_list: tuple[Atom, ...]) -> tuple[CrossingComposite, ...]:
    old = {(composite.top, composite.bottom, composite.positions) for composite in single_crossing_composites(atom_list)}
    return tuple(
        composite
        for composite in crossing_composites(atom_list)
        if (composite.top, composite.bottom, composite.positions) not in old
    )


def last_strand_permutations(order: Order) -> tuple[tuple[int, ...], ...]:
    # These are exactly the HI3 permutations involving the atom's last strand.
    out: list[tuple[int, ...]] = []
    seen = set()
    width = len(order)
    if width < 2:
        return ()
    for start in permutations(range(width)):
        positions = left_reidemeister_swaps(start)
        current = tuple(order[idx] for idx in start)
        strands = list(start)
        ok = True
        involves_last = False
        for pos in positions:
            if current[pos] == current[pos + 1]:
                ok = False
                break
            if strands[pos] == width - 1 or strands[pos + 1] == width - 1:
                involves_last = True
            current = swap_order(current, pos)
            strands[pos], strands[pos + 1] = strands[pos + 1], strands[pos]
        if ok and involves_last and current == order:
            key = (tuple(order[idx] for idx in start), positions)
            if key not in seen:
                seen.add(key)
                out.append(positions)
    return tuple(out)


def bottom_crossing_composites(atom_list: tuple[Atom, ...]) -> tuple[BottomCrossingComposite, ...]:
    out: list[BottomCrossingComposite] = []
    for atom in atom_list:
        bottom = EmbeddedAtom(atom, (), ())
        for positions in last_strand_permutations(bottom.domain):
            out.append(BottomCrossingComposite(bottom, positions))
    return tuple(sorted(out, key=lambda composite: (len(composite.bottom.domain), composite.bottom.domain, composite.bottom.codomain, composite.positions)))


def right_crossing_then_atom_composites(atom_list: tuple[Atom, ...]) -> tuple[RightCrossingThenAtomComposite, ...]:
    out: list[RightCrossingThenAtomComposite] = []
    for atom in atom_list:
        for right_label in atom.context.index_set:
            embedded = EmbeddedAtom(atom, (), (right_label,))
            if embedded.domain[-2] != embedded.domain[-1]:
                out.append(RightCrossingThenAtomComposite(embedded, right_label))
    return tuple(sorted(out, key=lambda composite: (len(composite.domain), composite.domain, composite.codomain)))


def common_middle_for_or2(atom_a: Atom, atom_b: Atom, overlap: int) -> tuple[Order, EmbeddedAtom, EmbeddedAtom] | None:
    if overlap < 1 or overlap > len(atom_a.domain) or overlap > len(atom_b.codomain):
        return None
    if atom_a.domain[-overlap:] != atom_b.codomain[:overlap]:
        return None
    middle = atom_a.domain + atom_b.codomain[overlap:]
    atom_a_embedding = EmbeddedAtom(atom_a, (), middle[len(atom_a.domain) :])
    atom_b_embedding = EmbeddedAtom(atom_b, middle[: len(atom_a.domain) - overlap], ())
    return middle, atom_a_embedding, atom_b_embedding


def overlapping_atom_composites(atom_list: tuple[Atom, ...]) -> tuple[OverlappingAtomComposite, ...]:
    return _all_overlapping_atom_composites(atom_list)


def direct_overlapping_atom_composites(atom_list: tuple[Atom, ...]) -> tuple[OverlappingAtomComposite, ...]:
    # OR2 with P equal to the identity.
    out: list[OverlappingAtomComposite] = []
    seen = set()
    for atom_a in atom_list:
        for atom_b in atom_list:
            for overlap in range(1, min(len(atom_a.domain), len(atom_b.codomain)) + 1):
                found = common_middle_for_or2(atom_a, atom_b, overlap)
                if found is None:
                    continue
                middle, atom_a_embedding, atom_b_embedding = found
                key = (middle, atom_a_embedding, atom_b_embedding)
                if key in seen:
                    continue
                seen.add(key)
                out.append(OverlappingAtomComposite(middle, atom_a_embedding, atom_b_embedding))

    for atom_a in atom_list:
        width = len(atom_a.domain)
        for atom_b in atom_list:
            middle = atom_b.codomain
            for pos in range(len(middle) - width + 1):
                if middle[pos : pos + width] != atom_a.domain:
                    continue
                atom_a_embedding = EmbeddedAtom(atom_a, middle[:pos], middle[pos + width :])
                atom_b_embedding = EmbeddedAtom(atom_b, (), ())
                key = (middle, atom_a_embedding, atom_b_embedding)
                if key in seen:
                    continue
                seen.add(key)
                out.append(OverlappingAtomComposite(middle, atom_a_embedding, atom_b_embedding))
    return tuple(sorted(out, key=lambda composite: (len(composite.middle), composite.middle, composite.domain, composite.codomain)))


def _all_overlapping_atom_composites(atom_list: tuple[Atom, ...]) -> tuple[OverlappingAtomComposite, ...]:
    # Add the OR2 cases where the overlap is mediated by a middle permutation.
    out = list(direct_overlapping_atom_composites(atom_list))
    seen = {(composite.atom_a, composite.atom_b, composite.positions) for composite in out}
    labels = atom_list[0].context.index_set if atom_list else ()

    for atom_a in atom_list:
        for atom_b in atom_list:
            max_width = len(atom_a.domain) + len(atom_b.codomain)
            for width in range(max(len(atom_a.domain), len(atom_b.codomain)), max_width + 1):
                for atom_b_pos in range(width - len(atom_b.codomain) + 1):
                    atom_b_block = range(atom_b_pos, atom_b_pos + len(atom_b.codomain))
                    for atom_a_pos in range(width - len(atom_a.domain) + 1):
                        atom_a_block = range(atom_a_pos, atom_a_pos + len(atom_a.domain))
                        if atom_a_block.stop > atom_b_block.stop:
                            continue
                        prefix_width = atom_a_block.stop - 1
                        if prefix_width < 0:
                            continue
                        for start in orders_with_block(labels, width, atom_b.codomain, atom_b_pos):
                            for target in orders_with_block(labels, width, atom_a.domain, atom_a_pos):
                                if Counter(start) != Counter(target):
                                    continue
                                target_source = target_to_source(start, target)
                                protected_source = target_source[atom_a_block.stop - 1]
                                if protected_source not in atom_b_block:
                                    continue
                                if not no_passive_middle_strands(start, target, atom_b_block, atom_a_block):
                                    continue
                                if not left_strands_interact(start, target, atom_b_block, atom_a_block):
                                    continue
                                positions = prefix_crossing_positions(start, target, prefix_width, {protected_source})
                                if positions is None:
                                    continue
                                atom_b_embedding = EmbeddedAtom(atom_b, start[:atom_b_pos], start[atom_b_block.stop :])
                                atom_a_embedding = EmbeddedAtom(atom_a, target[:atom_a_pos], target[atom_a_block.stop :])
                                key = (atom_a_embedding, atom_b_embedding, positions)
                                if key in seen:
                                    continue
                                seen.add(key)
                                out.append(OverlappingAtomComposite(target, atom_a_embedding, atom_b_embedding, positions))

    return tuple(sorted(out, key=lambda composite: (len(composite.middle), composite.middle, composite.domain, composite.codomain, composite.word())))


def additional_overlapping_atom_composites(atom_list: tuple[Atom, ...]) -> tuple[OverlappingAtomComposite, ...]:
    old = {(composite.atom_a, composite.atom_b, composite.positions) for composite in direct_overlapping_atom_composites(atom_list)}
    return tuple(
        composite
        for composite in overlapping_atom_composites(atom_list)
        if (composite.atom_a, composite.atom_b, composite.positions) not in old
    )


def rotate_right_strand_then_atom_composites(atom_list: tuple[Atom, ...]) -> tuple[RotateRightStrandThenAtomComposite, ...]:
    out: list[RotateRightStrandThenAtomComposite] = []
    for atom in atom_list:
        for right_label in atom.context.index_set:
            if right_label in atom.domain:
                continue
            embedded = EmbeddedAtom(atom, (right_label,), ())
            order = atom.domain + (right_label,)
            crossing_terms: list[str] = []
            for pos in range(len(atom.domain) - 1, -1, -1):
                crossing_terms.append(atom.context.tensor_embed(atom.context.crossing_symbol(order[pos : pos + 2]), order[:pos], order[pos + 2 :]))
                order = order[:pos] + (order[pos + 1], order[pos]) + order[pos + 2 :]
            out.append(RotateRightStrandThenAtomComposite(embedded, tuple(crossing_terms)))
    return tuple(sorted(out, key=lambda composite: (len(composite.domain), composite.domain, composite.codomain)))


def reidemeister_three_composites(context: ClassicalBPT) -> tuple[ReidemeisterThreeComposite, ...]:
    # Enumerate the OR4 three-crossing words for distinct labels.
    out: list[ReidemeisterThreeComposite] = []
    for labels in product(context.index_set, repeat=3):
        if len(set(labels)) != 3:
            continue
        order = tuple(labels)
        crossing_terms: list[str] = []
        for pos in (1, 0, 1):
            crossing_terms.append(context.tensor_embed(context.crossing_symbol(order[pos : pos + 2]), order[:pos], order[pos + 2 :]))
            order = order[:pos] + (order[pos + 1], order[pos]) + order[pos + 2 :]
        out.append(ReidemeisterThreeComposite(context, tuple(labels), tuple(crossing_terms)))
    return tuple(out)


def tensor_weight(context: ClassicalBPT, order: Order, tensor: Tensor):
    return context.sage_weight(context.to_sage(order, tensor))


def highest_rows(context: ClassicalBPT, domain: Order, codomain: Order) -> tuple[tuple[Tensor, Tensor], ...]:
    # Rows for the linear system: highest tensors of the same weight on both sides.
    domain_highest = context.highest_tensors(domain)
    codomain_highest = context.highest_tensors(codomain)
    return tuple(
        (domain_tensor, codomain_tensor)
        for domain_tensor in domain_highest
        for codomain_tensor in codomain_highest
        if tensor_weight(context, domain, domain_tensor) == tensor_weight(context, codomain, codomain_tensor)
    )


def map_vector(context: ClassicalBPT, domain: Order, codomain: Order, rows: tuple[tuple[Tensor, Tensor], ...], morphism) -> list:
    # Evaluate a crystal map on highest tensors and record its matrix column.
    values = []
    row_set = set(rows)
    for domain_tensor, codomain_tensor in rows:
        values.append(QQ(1) if morphism(domain_tensor) == codomain_tensor else QQ(0))
    for domain_tensor in {row[0] for row in rows}:
        image = morphism(domain_tensor)
        if image is not None and (domain_tensor, image) not in row_set:
            raise ValueError("map sends a highest tensor outside the codomain highest tensors")
    return values


def expand_in_bpt(composite):
    context = composite.context
    basis = context.bpt_basis(composite.domain, composite.codomain)
    if not basis:
        return ()
    rows = highest_rows(context, composite.domain, composite.codomain)
    columns = [map_vector(context, composite.domain, composite.codomain, rows, diagram.apply) for diagram in basis]
    rhs = vector(QQ, map_vector(context, composite.domain, composite.codomain, rows, composite.apply))
    coeffs = matrix(QQ, columns).transpose().solve_right(rhs)
    return tuple((coeff, diagram) for coeff, diagram in zip(coeffs, basis) if coeff)


def format_rhs(expansion) -> str:
    # Print a linear combination of BPT diagrams.
    if not expansion:
        return "0"
    pieces = []
    for coeff, diagram in expansion:
        sign = "-" if coeff < 0 else "+"
        size = -coeff if coeff < 0 else coeff
        term = diagram.word() if size == 1 else f"{size}*{diagram.word()}"
        pieces.append((sign, term))
    first_sign, first_term = pieces[0]
    out = f"- {first_term}" if first_sign == "-" else first_term
    for sign, term in pieces[1:]:
        out += f" {sign} {term}"
    return out


def relation_families(context: ClassicalBPT, atom_list: tuple[Atom, ...]):
    return {
        name: relation_family(context, atom_list, name)
        for name in ("HI1", "HI2", "HI3", "OR1", "OR2", "OR3", "OR4")
    }


def relation_family(context: ClassicalBPT, atom_list: tuple[Atom, ...], name: str):
    # Build only the requested relation family.
    if name == "HI1":
        return atom_composites(atom_list)
    if name == "HI2":
        return crossing_composites(atom_list)
    if name == "HI3":
        return bottom_crossing_composites(atom_list)
    if name == "OR1":
        return right_crossing_then_atom_composites(atom_list)
    if name == "OR2":
        return overlapping_atom_composites(atom_list)
    if name == "OR3":
        return rotate_right_strand_then_atom_composites(atom_list)
    if name == "OR4":
        return reidemeister_three_composites(context)
    raise ValueError(f"unknown relation family {name}")


def selected_family_names(name: str) -> tuple[str, ...]:
    if name == "all":
        return ("HI1", "HI2", "HI3", "OR1", "OR2", "OR3", "OR4")
    return (name,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate classical atom relations in the bottom-permutation-top basis.")
    parser.add_argument("cartan_type", help="A3, B4, C3, D5, ...")
    parser.add_argument("--max-atom-len", type=int, help="optional filter on atom domain length")
    parser.add_argument("--family", choices=("HI1", "HI2", "HI3", "OR1", "OR2", "OR3", "OR4", "all"), default="HI1")
    parser.add_argument("--max-middle-len", type=int)
    parser.add_argument("--show-zero", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.max_atom_len is not None and args.max_atom_len < 2:
        parser.error("--max-atom-len must be at least 2")

    try:
        context = ClassicalBPT(parse_cartan_type(args.cartan_type))
    except ValueError as error:
        parser.error(str(error))

    atom_list = atoms(context, args.max_atom_len)
    families = {}
    for name in selected_family_names(args.family):
        family = relation_family(context, atom_list, name)
        if name == "HI1" and args.max_middle_len is not None:
            family = tuple(composite for composite in family if len(composite.middle) <= args.max_middle_len)
        families[name] = family

    print(f"{args.cartan_type.upper()} atoms: {len(atom_list)}")
    for name in selected_family_names(args.family):
        print(f"{name} relations: {len(families[name])}")
    if args.count_only:
        return

    composites = tuple(item for family in families.values() for item in family)
    shown = 0
    for composite in composites:
        expansion = expand_in_bpt(composite)
        if not expansion and not args.show_zero:
            continue
        shown += 1
        print(f"{shown}. {composite.family} {composite.word()} : {fmt_order(composite.domain)} -> {fmt_order(composite.codomain)}")
        print(f"   = {format_rhs(expansion)}")
        if args.limit is not None and shown >= args.limit:
            break


if __name__ == "__main__":
    main()
