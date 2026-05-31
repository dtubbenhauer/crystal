"""Minimal G2 bottom-permutation-top basis code.

Run with Sage's Python, for example:

    sage -python g2_bpt.py 112 21
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import product

from sage.all import CartanType, QQ, crystals, matrix


Weight = tuple[int, int]
Order = tuple[int, ...]
Tensor = tuple[object, ...]
INDEX_SET = (1, 2)
ROOT_WEIGHT = {1: (1, 0), 2: (0, 1)}


def add_weight(left: Weight, right: Weight) -> Weight: return left[0] + right[0], left[1] + right[1]


def fmt_order(order: Order) -> str: return "(0)" if not order else "(" + ",".join(map(str, order)) + ")"


def label_word(order: Order) -> str: return "0" if not order else "".join(map(str, order))


def parse_word(text: str) -> Order:
    # Read a boundary word such as 112, 1,2,1, or 0.
    cleaned = text.replace(",", "").replace(" ", "")
    if cleaned == "0":
        return ()
    if not cleaned:
        raise argparse.ArgumentTypeError("word must contain at least one label")
    if any(char not in "12" for char in cleaned):
        raise argparse.ArgumentTypeError("only labels 1 and 2 are allowed")
    return tuple(int(char) for char in cleaned)


CRYSTAL = {
    1: crystals.LSPaths(CartanType(["G", 2]), [1, 0]),
    2: crystals.LSPaths(CartanType(["G", 2]), [0, 1]),
}


def sage_parent(order: Order):
    # Build the Sage crystal for a boundary word.
    if not order:
        return None
    # Convert our tensor order to Sage's reversed tensor-factor convention.
    factors = [CRYSTAL[label] for label in reversed(order)]
    return factors[0] if len(factors) == 1 else crystals.TensorProduct(*factors)


def to_sage(order: Order, tensor: Tensor):
    # Convert from the paper's tensor order to Sage's tensor element.
    if not order:
        return ()
    if len(order) == 1:
        return tensor[0]
    return sage_parent(order)(*reversed(tensor))


def from_sage(order: Order, element) -> Tensor:
    # Convert a Sage tensor element back to the paper's tensor order.
    if not order:
        return ()
    if len(order) == 1:
        return (element,)
    return tuple(reversed(tuple(element)))


def basis_for_word(order: Order) -> list[Tensor]:
    if not order:
        return [()]
    return list(product(*(CRYSTAL[label] for label in order)))


def sage_weight(element) -> Weight: return (0, 0) if element == () else tuple(int(entry) for entry in element.weight().to_vector())


def highest_tensors(order: Order) -> tuple[Tensor, ...]:
    # Highest-weight tensors are the test vectors for the independence check.
    if not order:
        return ((),)
    return tuple(from_sage(order, element) for element in sage_parent(order).highest_weight_vectors())


def is_highest_in(element, selected: set) -> bool: return all((image := element.e(i)) is None or image not in selected for i in INDEX_SET)


def highest_in(order: Order, highest_weight: Weight, selected_tensors=None):
    # Find the highest vector that determines the relevant summand.
    if not order:
        if highest_weight == (0, 0):
            return ()
        raise ValueError(f"{fmt_order(order)}: no highest element of weight {highest_weight}")

    if selected_tensors is None:
        candidates = sage_parent(order).highest_weight_vectors()
    else:
        selected = set(to_sage(order, tensor) for tensor in selected_tensors)
        candidates = [element for element in selected if is_highest_in(element, selected)]

    matches = [element for element in candidates if sage_weight(element) == highest_weight]
    if len(matches) != 1:
        raise ValueError(f"{fmt_order(order)}: expected one highest vector of weight {highest_weight}, got {len(matches)}")
    return matches[0]


def summand_tensors(order: Order, highest_weight: Weight) -> set[Tensor]:
    # Identify a summand so branching rules use the right copy in multiplicity cases.
    highest = highest_in(order, highest_weight)
    if not order:
        return {()}
    parent = sage_parent(order)
    identity = parent.crystal_morphism([highest], codomain=parent, generators=[highest], check=False)
    out: set[Tensor] = set()
    for tensor in basis_for_word(order):
        try:
            identity(to_sage(order, tensor))
        except ValueError:
            continue
        out.add(tensor)
    return out


@dataclass(frozen=True)
class BranchingRule:
    # One local branching atom in the recursive construction.
    delta: Weight
    new_label: int
    old_suffix: Order
    codomain_order: Order
    old_suffix_weight: Weight
    min_parent_weight: Weight = (0, 0)

    @property
    def name(self) -> str: return f"{self.new_label}:{label_word(self.old_suffix)}>{label_word(self.codomain_order)}"

    @property
    def domain_order(self) -> Order: return self.old_suffix + (self.new_label,)

    @property
    def branch_weight(self) -> Weight: return add_weight(self.old_suffix_weight, self.delta)

    def applies(self, parent_weight: Weight) -> bool:
        return all(have >= need for have, need in zip(parent_weight, self.min_parent_weight))


BRANCHING_RULES = (
    BranchingRule((1, 0), 1, (), (1,), (0, 0)),
    BranchingRule((-1, 1), 1, (1,), (2,), (1, 0), (1, 0)),
    BranchingRule((2, -1), 1, (2,), (1, 1), (0, 1), (0, 1)),
    BranchingRule((0, 0), 1, (1,), (1,), (1, 0), (1, 0)),
    BranchingRule((1, -1), 1, (2,), (1,), (0, 1), (0, 1)),
    BranchingRule((-1, 0), 1, (1,), (), (1, 0), (1, 0)),
    BranchingRule((-2, 1), 1, (1, 1), (2,), (2, 0), (2, 0)),
    BranchingRule((0, 1), 2, (), (2,), (0, 0)),
    BranchingRule((3, -1), 2, (2,), (1, 1, 1), (0, 1), (0, 1)),
    BranchingRule((3, -2), 2, (2, 2), (1, 1, 1), (0, 2), (0, 2)),
    BranchingRule((2, -1), 2, (2,), (1, 1), (0, 1), (0, 1)),
    BranchingRule((1, 0), 2, (1,), (1, 1), (1, 0), (1, 0)),
    BranchingRule((1, -1), 2, (1, 2), (1, 1), (1, 1), (1, 1)),
    BranchingRule((0, 0), 2, (1,), (1,), (1, 0), (1, 0)),
    BranchingRule((0, 0), 2, (2,), (2,), (0, 1), (0, 1)),
    BranchingRule((0, -1), 2, (2,), (), (0, 1), (0, 1)),
    BranchingRule((-1, 1), 2, (1, 1), (1, 2), (2, 0), (2, 0)),
    BranchingRule((-1, 0), 2, (1, 1), (1,), (2, 0), (2, 0)),
    BranchingRule((-2, 1), 2, (1, 1), (2,), (2, 0), (2, 0)),
    BranchingRule((-3, 2), 2, (1, 1, 1), (2, 2), (3, 0), (3, 0)),
    BranchingRule((-3, 1), 2, (1, 1, 1), (2,), (3, 0), (3, 0)),
)
BRANCHING_RULES_BY_NAME = {rule.name: rule for rule in BRANCHING_RULES}


@dataclass(frozen=True)
class Step:
    # One layer in a branch word: either a crossing or an atom.
    kind: str
    pos: int
    branching_name: str = ""


@dataclass(frozen=True)
class BranchNode:
    # One summand reached while recursively building a boundary word.
    highest_weight: Weight
    order: Order
    steps: tuple[Step, ...]
    apply: object


def branching_domain_tensors(branching_rule: BranchingRule) -> list[Tensor]:
    # Restrict a branching rule to the summand selected by its old suffix.
    domain_basis = basis_for_word(branching_rule.domain_order)
    if not branching_rule.old_suffix:
        return domain_basis
    prefixes = summand_tensors(branching_rule.old_suffix, branching_rule.old_suffix_weight)
    width = len(branching_rule.old_suffix)
    return [tensor for tensor in domain_basis if tensor[:width] in prefixes]


def apply_branching_rule(branching_rule: BranchingRule, tensor: Tensor, inverse: bool = False) -> Tensor | None:
    # Apply the branching map used at one vertex of a diagram.
    selected = branching_domain_tensors(branching_rule)
    domain_highest = highest_in(branching_rule.domain_order, branching_rule.branch_weight, selected)
    codomain_highest = highest_in(branching_rule.codomain_order, branching_rule.branch_weight)
    domain_parent = sage_parent(branching_rule.domain_order)

    try:
        if inverse:
            if not branching_rule.codomain_order:
                return from_sage(branching_rule.domain_order, domain_highest) if tensor == () else None
            sage_map = sage_parent(branching_rule.codomain_order).crystal_morphism(
                [domain_highest],
                codomain=domain_parent,
                generators=[codomain_highest],
                check=False,
            )
            return from_sage(branching_rule.domain_order, sage_map(to_sage(branching_rule.codomain_order, tensor)))

        if not branching_rule.codomain_order:
            membership = domain_parent.crystal_morphism(
                [domain_highest],
                codomain=domain_parent,
                generators=[domain_highest],
                check=False,
            )
            membership(to_sage(branching_rule.domain_order, tensor))
            return ()
        sage_map = domain_parent.crystal_morphism(
            [codomain_highest],
            codomain=sage_parent(branching_rule.codomain_order),
            generators=[domain_highest],
            check=False,
        )
        return from_sage(branching_rule.codomain_order, sage_map(to_sage(branching_rule.domain_order, tensor)))
    except ValueError:
        return None


def sage_commutor(pair: Order, tensor: Tensor) -> Tensor:
    if pair not in {(1, 2), (2, 1)}:
        raise ValueError("crossing requested on equal labels")
    codomain_order = (pair[1], pair[0])
    # Compute the commutor B_a x B_b -> B_b x B_a using Sage's Lusztig involution.
    seed = tuple(x.lusztig_involution() for x in reversed(tensor))
    return from_sage(codomain_order, to_sage(codomain_order, seed).lusztig_involution())


def id_symbol(label: int) -> str: return f"id{label}"


def branching_symbol(branching_rule: BranchingRule, flipped: bool = False) -> str:
    if not branching_rule.old_suffix:
        return "id"
    symbol = f"[{label_word(branching_rule.domain_order)}->{label_word(branching_rule.codomain_order)}]"
    return f"bar({symbol})" if flipped else symbol


def crossing_symbol(pair: Order) -> str:
    if pair not in {(1, 2), (2, 1)}:
        raise ValueError("crossing requested on equal labels")
    return f"beta[{label_word(pair)}]"


def tensor_embed(core: str, left: Order, right: Order) -> str:
    if core == "id":
        return "id"
    factors = [id_symbol(label) for label in left] + [core] + [id_symbol(label) for label in right]
    return " x ".join(factors)


def compose_terms(execution_terms: list[str]) -> str:
    terms = [term for term in execution_terms if term != "id"]
    if not terms:
        return "id"
    return " o ".join(reversed([parenthesize(term) for term in terms]))


def parenthesize(term: str) -> str:
    if term.startswith("(") and term.endswith(")"):
        return term
    if " o " in term or " x " in term:
        return f"({term})"
    return term


def step_word(step: Step, current_order: Order, top: bool = False) -> str:
    pos = step.pos - 1
    if step.kind == "cross":
        return tensor_embed(crossing_symbol(current_order[pos : pos + 2]), current_order[:pos], current_order[pos + 2 :])
    branching_rule = BRANCHING_RULES_BY_NAME[step.branching_name]
    width = len(branching_rule.codomain_order if top else branching_rule.domain_order)
    return tensor_embed(branching_symbol(branching_rule, flipped=top), current_order[:pos], current_order[pos + width :])


def bottom_word(boundary: Order, steps: tuple[Step, ...]) -> str:
    # Print the atom/crossing word for a bottom branch.
    order = boundary
    terms: list[str] = []
    for step in steps:
        terms.append(step_word(step, order))
        order = apply_step_to_order(order, step)
    return compose_terms(terms)


def top_word(boundary: Order, steps: tuple[Step, ...], domain_order: Order) -> str:
    # Build the printed word for the top half of a diagram.
    trace = order_trace(boundary, steps)
    order = domain_order
    terms: list[str] = []
    for idx in range(len(steps) - 1, -1, -1):
        step = steps[idx]
        terms.append(step_word(step, order, top=True))
        order = trace[idx]
    return compose_terms(terms)


def apply_commutor(outputs: list[object], order: list[int], pos: int) -> None:
    # Apply a crossing to adjacent tensor factors.
    pair = tuple(order[pos : pos + 2])
    image = sage_commutor(pair, tuple(outputs[pos : pos + 2]))
    outputs[pos : pos + 2] = list(image)
    order[pos : pos + 2] = list(reversed(pair))


def apply_step_to_order(order: Order, step: Step) -> Order:
    # Track how a branch step changes the boundary word.
    pos = step.pos - 1
    if step.kind == "cross":
        return order[:pos] + (order[pos + 1], order[pos]) + order[pos + 2 :]
    branching_rule = BRANCHING_RULES_BY_NAME[step.branching_name]
    width = len(branching_rule.domain_order)
    if order[pos : pos + width] != branching_rule.domain_order:
        raise ValueError(f"{step.branching_name}: domain mismatch")
    return order[:pos] + branching_rule.codomain_order + order[pos + width :]


def execute_steps(tensor: Tensor, start_order: Order, steps: tuple[Step, ...]) -> Tensor | None:
    # Apply a whole branch word to a tensor.
    outputs = list(tensor)
    order = list(start_order)
    for step in steps:
        pos = step.pos - 1
        if step.kind == "cross":
            if tuple(order[pos : pos + 2]) not in {(1, 2), (2, 1)}:
                return None
            apply_commutor(outputs, order, pos)
            continue

        branching_rule = BRANCHING_RULES_BY_NAME[step.branching_name]
        width = len(branching_rule.domain_order)
        if tuple(order[pos : pos + width]) != branching_rule.domain_order:
            return None
        # Embed the branching map in the full tensor product by changing only this block.
        image = apply_branching_rule(branching_rule, tuple(outputs[pos : pos + width]))
        if image is None:
            return None
        outputs[pos : pos + width] = list(image)
        order[pos : pos + width] = list(branching_rule.codomain_order)
    return tuple(outputs)


def route_suffix(order: Order, suffix: Order) -> tuple[tuple[int, ...], Order] | None:
    # Choose the deterministic crossings needed before applying a branching rule.
    if not suffix:
        return (), order
    if len(suffix) > len(order):
        return None
    if any(order.count(label) < suffix.count(label) for label in set(suffix)):
        return None

    labels = list(order)
    swaps: list[int] = []
    # Fill suffix positions from right to left using the rightmost available label.
    for desired_idx in range(len(order) - 1, len(order) - len(suffix) - 1, -1):
        desired_label = suffix[desired_idx - (len(order) - len(suffix))]
        found_idx = None
        for idx in range(desired_idx, -1, -1):
            if labels[idx] == desired_label:
                found_idx = idx
                break
        if found_idx is None:
            return None

        for idx in range(found_idx, desired_idx):
            if labels[idx] == labels[idx + 1]:
                raise ValueError("routing would cross equal labels")
            labels[idx], labels[idx + 1] = labels[idx + 1], labels[idx]
            swaps.append(idx + 1)

    routed = tuple(labels)
    if routed[-len(suffix) :] != suffix:
        raise ValueError("suffix routing failed")
    return tuple(swaps), routed


def build_branching(labels: Order) -> tuple[tuple[BranchNode, ...], ...]:
    # Construct the branching tree for one boundary word.
    # A branching rule contributes after route_suffix moves its old_suffix next to the new label.
    if any(label not in {1, 2} for label in labels):
        raise ValueError("boundary word must use labels 1,2")
    if not labels:
        return ((BranchNode((0, 0), (), (), lambda tensor: tensor),),)

    root = BranchNode(ROOT_WEIGHT[labels[0]], (labels[0],), (), lambda tensor: tensor)
    stages: list[tuple[BranchNode, ...]] = [(root,)]

    for stage, new_label in enumerate(labels[1:], start=2):
        current_stage: list[BranchNode] = []
        for parent in stages[-1]:
            for branching_rule in (r for r in BRANCHING_RULES if r.new_label == new_label):
                if not branching_rule.applies(parent.highest_weight):
                    continue
                routed = route_suffix(parent.order, branching_rule.old_suffix)
                if routed is None:
                    continue
                swaps, _ = routed
                offset = len(parent.order) - len(branching_rule.old_suffix)
                extension_steps = tuple(Step("cross", pos) for pos in swaps) + (
                    Step("branch", offset + 1, branching_rule.name),
                )

                def apply(tensor, parent=parent, new_label=new_label, extension_steps=extension_steps):
                    # Apply the parent map to the old factors, then the new branching extension.
                    prefix = parent.apply(tensor[:-1])
                    if prefix is None:
                        return None
                    return execute_steps(prefix + (tensor[-1],), parent.order + (new_label,), extension_steps)

                current_stage.append(
                    BranchNode(
                        add_weight(parent.highest_weight, branching_rule.delta),
                        order_trace(parent.order + (new_label,), extension_steps)[-1],
                        parent.steps + extension_steps,
                        apply,
                    )
                )
        stages.append(tuple(current_stage))
    return tuple(stages)


def order_trace(boundary: Order, steps: tuple[Step, ...]) -> tuple[Order, ...]:
    # Record order changes so the same branch can be traversed backward.
    trace = [boundary]
    order = boundary
    for step in steps:
        order = apply_step_to_order(order, step)
        trace.append(order)
    return tuple(trace)


def apply_top(boundary: Order, steps: tuple[Step, ...], domain_order: Order, tensor: Tensor) -> Tensor | None:
    # Apply the top half of a diagram: the inverse of a codomain branch.
    outputs = list(tensor)
    current_order = domain_order
    trace = order_trace(boundary, steps)
    for idx in range(len(steps) - 1, -1, -1):
        step = steps[idx]
        before = trace[idx]
        after = trace[idx + 1]
        if current_order != after:
            raise ValueError("top order mismatch")
        pos = step.pos - 1
        if step.kind == "cross":
            order = list(current_order)
            apply_commutor(outputs, order, pos)
            current_order = tuple(order)
            continue

        branching_rule = BRANCHING_RULES_BY_NAME[step.branching_name]
        width = len(branching_rule.codomain_order)
        image = apply_branching_rule(branching_rule, tuple(outputs[pos : pos + width]), inverse=True)
        if image is None:
            return None
        outputs[pos : pos + width] = list(image)
        current_order = before
    if current_order != boundary:
        raise ValueError("top boundary mismatch")
    return tuple(outputs)


def permutation_swaps(domain_order: Order, codomain_order: Order) -> tuple[int, ...]:
    # Choose the crossing word for the middle permutation in a BPT diagram.
    labels = list(domain_order)
    swaps: list[int] = []
    if sorted(labels) != sorted(codomain_order):
        raise ValueError("orders have different multisets")
    for idx, desired_label in enumerate(codomain_order):
        if labels[idx] == desired_label:
            continue
        swap_idx = next(pos for pos in range(idx + 1, len(labels)) if labels[pos] == desired_label)
        for pos in range(swap_idx - 1, idx - 1, -1):
            if labels[pos] == labels[pos + 1]:
                raise ValueError("equal-label crossings are not used")
            labels[pos], labels[pos + 1] = labels[pos + 1], labels[pos]
            swaps.append(pos + 1)
    return tuple(swaps)


def commute_to_order(tensor: Tensor, domain_order: Order, codomain_order: Order) -> Tensor:
    # Apply the middle permutation to a tensor.
    outputs = list(tensor)
    order = list(domain_order)
    for swap_pos in permutation_swaps(domain_order, codomain_order):
        apply_commutor(outputs, order, swap_pos - 1)
    return tuple(outputs)


def permutation_word(domain_order: Order, codomain_order: Order) -> str:
    # Print the middle permutation word.
    order = list(domain_order)
    terms: list[str] = []
    for swap_pos in permutation_swaps(domain_order, codomain_order):
        pos = swap_pos - 1
        current_order = tuple(order)
        terms.append(tensor_embed(crossing_symbol(current_order[pos : pos + 2]), current_order[:pos], current_order[pos + 2 :]))
        order[pos], order[pos + 1] = order[pos + 1], order[pos]
    return compose_terms(terms)


@dataclass(frozen=True)
class Diagram:
    # A bottom-perm-top basis diagram.
    domain: Order
    codomain: Order
    bottom: BranchNode
    top: BranchNode

    @property
    def weight(self) -> Weight: return self.bottom.highest_weight

    def word(self) -> str:
        return compose_terms(
            [
                bottom_word(self.domain, self.bottom.steps),
                permutation_word(self.bottom.order, self.top.order),
                top_word(self.codomain, self.top.steps, self.top.order),
            ]
        )

    def apply(self, tensor: Tensor) -> Tensor | None:
        # Evaluate the full bottom-perm-top diagram as a crystal map.
        middle = self.bottom.apply(tensor)
        if middle is None:
            return None
        middle = commute_to_order(middle, self.bottom.order, self.top.order)
        return apply_top(self.codomain, self.top.steps, self.top.order, middle)


def bpt_basis(domain: Order, codomain: Order) -> tuple[Diagram, ...]:
    # Generate the BPT diagrams by matching domain and codomain branching summands.
    bottom_nodes = build_branching(domain)[-1]
    top_nodes = build_branching(codomain)[-1]
    return tuple(
        Diagram(domain, codomain, bottom, top)
        for bottom in bottom_nodes
        for top in top_nodes
        if bottom.highest_weight == top.highest_weight
    )


def decomposition_counts(word: Order) -> Counter[Weight]:
    if not word:
        return Counter({(0, 0): 1})
    return Counter(sage_weight(element) for element in sage_parent(word).highest_weight_vectors())


def expected_hom_dimension(domain: Order, codomain: Order) -> int:
    # Compute the Hom dimension from decomposition multiplicities.
    domain_counts = decomposition_counts(domain)
    codomain_counts = decomposition_counts(codomain)
    return sum(domain_counts[weight] * codomain_counts[weight] for weight in domain_counts.keys() & codomain_counts.keys())


def independence_rank(domain: Order, codomain: Order, diagrams: tuple[Diagram, ...]) -> int:
    # Check linear independence of the diagrams using highest-weight tensors.
    domain_highest = highest_tensors(domain)
    codomain_highest = highest_tensors(codomain)
    rows = {(s, t): idx for idx, (s, t) in enumerate(product(domain_highest, codomain_highest))}
    mat = matrix(QQ, len(rows), len(diagrams), sparse=True)
    for col, diagram in enumerate(diagrams):
        for domain_tensor in domain_highest:
            image = diagram.apply(domain_tensor)
            if image is None:
                continue
            if (domain_tensor, image) not in rows:
                raise ValueError("a diagram sent a highest tensor outside the codomain highest tensors")
            mat[rows[(domain_tensor, image)], col] = 1
    return mat.rank()


def main() -> None:
    parser = argparse.ArgumentParser(description="List the G2 bottom-permutation-top basis.")
    parser.add_argument("domain", type=parse_word)
    parser.add_argument("codomain", type=parse_word)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--check-independence", action="store_true")
    args = parser.parse_args()

    basis = bpt_basis(args.domain, args.codomain)
    expected = expected_hom_dimension(args.domain, args.codomain)
    print(f"{fmt_order(args.domain)} -> {fmt_order(args.codomain)}")
    print(f"basis diagrams: {len(basis)}")
    print(f"expected Hom dimension: {expected}")
    if args.check_independence:
        rank = independence_rank(args.domain, args.codomain, basis)
        print(f"highest-weight rank: {rank}")
        print(f"independent: {rank == len(basis)}")
    if args.count_only:
        return
    for idx, diagram in enumerate(basis, start=1):
        print(f"{idx}. wt={diagram.weight} {diagram.word()}")


if __name__ == "__main__":
    main()
