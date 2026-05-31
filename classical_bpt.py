"""Atom-based bottom-permutation-top bases for classical types A, B, C, D.

Run with Sage's Python, for example:

    sage -python classical_bpt.py A3 12 21
    sage -python classical_bpt.py C3 1,2,1 3 --check-independence
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from itertools import product

from sage.all import CartanType, QQ, crystals, matrix


Weight = tuple[int, ...]
Order = tuple[int, ...]
Tensor = tuple[object, ...]


def add_weight(left: Weight, right: Weight) -> Weight:
    return tuple(a + b for a, b in zip(left, right))


def subtract_weight(left: Weight, right: Weight) -> Weight:
    return tuple(a - b for a, b in zip(left, right))


def zero_weight(rank: int) -> Weight:
    return (0,) * rank


def fundamental_weight(rank: int, label: int) -> Weight:
    weight = [0] * rank
    weight[label - 1] = 1
    return tuple(weight)


def canonical_order(weight: Weight) -> Order:
    # Write a dominant weight as the corresponding ordered word of labels.
    order: list[int] = []
    for label, multiplicity in enumerate(weight, start=1):
        order.extend([label] * multiplicity)
    return tuple(order)


def order_weight(rank: int, order: Order) -> Weight:
    weight = [0] * rank
    for label in order:
        weight[label - 1] += 1
    return tuple(weight)


def label_word(order: Order) -> str:
    if not order:
        return "0"
    if all(label < 10 for label in order):
        return "".join(map(str, order))
    return ",".join(map(str, order))


def fmt_order(order: Order) -> str:
    return "(0)" if not order else "(" + ",".join(map(str, order)) + ")"


def fmt_weight(weight: Weight) -> str:
    return "(" + ",".join(map(str, weight)) + ")"


def parse_cartan_type(text: str):
    # Read a Cartan type name such as A3, B4, C3, or D5.
    match = re.fullmatch(r"([ABCDabcd])\s*([1-9][0-9]*)", text.strip())
    if not match:
        raise ValueError("Cartan type must look like A3, B4, C3, or D5")
    kind, rank_text = match.groups()
    kind = kind.upper()
    rank = int(rank_text)
    if kind == "A" and rank < 1:
        raise ValueError("type A needs rank at least 1")
    if kind in {"B", "C"} and rank < 2:
        raise ValueError(f"type {kind} needs rank at least 2")
    if kind == "D" and rank < 4:
        raise ValueError("type D needs rank at least 4")
    return CartanType([kind, rank])


def parse_word(text: str, rank: int) -> Order:
    # Read a boundary word, using commas when labels have more than one digit.
    cleaned = text.strip()
    if cleaned == "0":
        return ()
    if not cleaned:
        raise ValueError("word must contain at least one label")
    if "," in cleaned or " " in cleaned:
        labels = [int(part) for part in re.split(r"[\s,]+", cleaned) if part]
    elif rank <= 9:
        labels = [int(char) for char in cleaned]
    else:
        raise ValueError("for rank at least 10, separate labels with commas")
    bad = [label for label in labels if label < 1 or label > rank]
    if bad:
        raise ValueError(f"labels must lie between 1 and {rank}")
    return tuple(labels)


@dataclass(frozen=True)
class BranchingRule:
    # One local branching atom in the recursive construction.
    name: str
    new_label: int
    old_suffix: Order
    codomain_order: Order
    branch_weight: Weight
    domain_highest: Tensor

    @property
    def domain_order(self) -> Order:
        return self.old_suffix + (self.new_label,)


@dataclass(frozen=True)
class BranchChoice:
    # A branching rule together with the resulting highest weight.
    rule: BranchingRule
    child_weight: Weight


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


@dataclass(frozen=True)
class Diagram:
    # A bottom-perm-top basis diagram.
    context: "ClassicalBPT"
    domain: Order
    codomain: Order
    bottom: BranchNode
    top: BranchNode

    @property
    def weight(self) -> Weight:
        return self.bottom.highest_weight

    def word(self) -> str:
        return self.context.compose_terms(
            [
                self.context.bottom_word(self.domain, self.bottom.steps),
                self.context.permutation_word(self.bottom.order, self.top.order),
                self.context.top_word(self.codomain, self.top.steps, self.top.order),
            ]
        )

    def apply(self, tensor: Tensor) -> Tensor | None:
        # Evaluate the diagram as bottom branch, permutation, then inverse top branch.
        middle = self.bottom.apply(tensor)
        if middle is None:
            return None
        middle = self.context.commute_to_order(middle, self.bottom.order, self.top.order)
        return self.context.apply_top(self.codomain, self.top.steps, self.top.order, middle)


class ClassicalBPT:
    # Holds the crystals and atom rules for one classical Cartan type.
    def __init__(self, cartan_type):
        self.cartan_type = cartan_type
        self.rank = cartan_type.rank()
        self.index_set = tuple(range(1, self.rank + 1))
        self.fundamental_crystals = {
            label: crystals.LSPaths(cartan_type, list(fundamental_weight(self.rank, label)))
            for label in self.index_set
        }
        self.branching_rules: dict[str, BranchingRule] = {}

    def sage_parent(self, order: Order):
        if not order:
            return None
        # Sage uses the opposite tensor-factor convention from the one used here.
        factors = [self.fundamental_crystals[label] for label in reversed(order)]
        return factors[0] if len(factors) == 1 else crystals.TensorProduct(*factors)

    def to_sage(self, order: Order, tensor: Tensor):
        if not order:
            return ()
        if len(order) == 1:
            return tensor[0]
        return self.sage_parent(order)(*reversed(tensor))

    def from_sage(self, order: Order, element) -> Tensor:
        if not order:
            return ()
        if len(order) == 1:
            return (element,)
        return tuple(reversed(tuple(element)))

    def sage_weight(self, element) -> Weight:
        if element == ():
            return zero_weight(self.rank)
        return tuple(int(entry) for entry in element.weight().to_vector())

    def highest_tensors(self, order: Order) -> tuple[Tensor, ...]:
        if not order:
            return ((),)
        return tuple(self.from_sage(order, element) for element in self.sage_parent(order).highest_weight_vectors())

    def highest_in(self, order: Order, highest_weight: Weight):
        if not order:
            if highest_weight == zero_weight(self.rank):
                return ()
            raise ValueError(f"{fmt_order(order)}: no highest element of weight {fmt_weight(highest_weight)}")
        matches = [
            element
            for element in self.sage_parent(order).highest_weight_vectors()
            if self.sage_weight(element) == highest_weight
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{fmt_order(order)}: expected one highest element of weight {fmt_weight(highest_weight)}, "
                f"got {len(matches)}"
            )
        return matches[0]

    def element_epsilon(self, element) -> Weight:
        return tuple(int(element.epsilon(i)) for i in self.index_set)

    def element_phi(self, element) -> Weight:
        return tuple(int(element.phi(i)) for i in self.index_set)

    def top_highest_tensor(self, order: Order) -> Tensor:
        if not order:
            return ()
        highest = self.highest_in(order, order_weight(self.rank, order))
        return self.from_sage(order, highest)

    def atom_rules_for_label(self, new_label: int) -> tuple[BranchingRule, ...]:
        # Definition 2B.8: one atom for each element of the fundamental crystal.
        raw = []
        for element in self.fundamental_crystals[new_label]:
            epsilon = self.element_epsilon(element)
            phi = self.element_phi(element)
            old_suffix = canonical_order(epsilon)
            codomain_order = canonical_order(phi)
            raw.append((old_suffix, codomain_order, phi, self.top_highest_tensor(old_suffix) + (element,)))

        counts = Counter((old_suffix, codomain_order) for old_suffix, codomain_order, _, _ in raw)
        seen = Counter()
        rules: list[BranchingRule] = []
        for old_suffix, codomain_order, phi, domain_highest in raw:
            domain_order = old_suffix + (new_label,)
            key = (old_suffix, codomain_order)
            seen[key] += 1
            base_name = f"{label_word(domain_order)}>{label_word(codomain_order)}"
            name = base_name if counts[key] == 1 else f"{base_name}#{seen[key]}"
            if name not in self.branching_rules:
                self.branching_rules[name] = BranchingRule(
                    name,
                    new_label,
                    old_suffix,
                    codomain_order,
                    phi,
                    domain_highest,
                )
            rules.append(self.branching_rules[name])
        return tuple(rules)

    def all_atom_rules(self) -> tuple[BranchingRule, ...]:
        # List all atoms for the fundamental labels of this type.
        return tuple(rule for label in self.index_set for rule in self.atom_rules_for_label(label))

    def rules_from(self, parent_weight: Weight, new_label: int) -> tuple[BranchChoice, ...]:
        # Use exactly those atoms whose epsilon-word fits inside the current highest weight.
        choices: list[BranchChoice] = []
        for rule in self.atom_rules_for_label(new_label):
            epsilon = order_weight(self.rank, rule.old_suffix)
            if any(have < need for have, need in zip(parent_weight, epsilon)):
                continue
            child_weight = add_weight(subtract_weight(parent_weight, epsilon), rule.branch_weight)
            choices.append(BranchChoice(rule, child_weight))
        return tuple(choices)

    def apply_branching_rule(self, branching_rule: BranchingRule, tensor: Tensor, inverse: bool = False) -> Tensor | None:
        # Sage builds the unique crystal morphism from the selected highest vector.
        domain_parent = self.sage_parent(branching_rule.domain_order)
        domain_highest = self.to_sage(branching_rule.domain_order, branching_rule.domain_highest)
        codomain_highest = self.highest_in(branching_rule.codomain_order, branching_rule.branch_weight)

        try:
            if inverse:
                if not branching_rule.codomain_order:
                    return branching_rule.domain_highest if tensor == () else None
                sage_map = self.sage_parent(branching_rule.codomain_order).crystal_morphism(
                    [domain_highest],
                    codomain=domain_parent,
                    generators=[codomain_highest],
                    check=False,
                )
                return self.from_sage(
                    branching_rule.domain_order,
                    sage_map(self.to_sage(branching_rule.codomain_order, tensor)),
                )

            if not branching_rule.codomain_order:
                identity = domain_parent.crystal_morphism(
                    [domain_highest],
                    codomain=domain_parent,
                    generators=[domain_highest],
                    check=False,
                )
                identity(self.to_sage(branching_rule.domain_order, tensor))
                return ()
            sage_map = domain_parent.crystal_morphism(
                [codomain_highest],
                codomain=self.sage_parent(branching_rule.codomain_order),
                generators=[domain_highest],
                check=False,
            )
            return self.from_sage(
                branching_rule.codomain_order,
                sage_map(self.to_sage(branching_rule.domain_order, tensor)),
            )
        except ValueError:
            return None

    def sage_commutor(self, pair: Order, tensor: Tensor) -> Tensor:
        # Apply Sage's crystal commutor to two adjacent distinct labels.
        if pair[0] == pair[1]:
            raise ValueError("crossing requested on equal labels")
        codomain_order = (pair[1], pair[0])
        seed = tuple(x.lusztig_involution() for x in reversed(tensor))
        return self.from_sage(codomain_order, self.to_sage(codomain_order, seed).lusztig_involution())

    def id_symbol(self, label: int) -> str:
        return f"id{label}"

    def branching_symbol(self, branching_rule: BranchingRule, flipped: bool = False) -> str:
        if not branching_rule.old_suffix:
            return "id"
        symbol = f"[{label_word(branching_rule.domain_order)}->{label_word(branching_rule.codomain_order)}]"
        if "#" in branching_rule.name:
            symbol = f"{symbol}{branching_rule.name[branching_rule.name.index('#'):]}"
        return f"bar({symbol})" if flipped else symbol

    def crossing_symbol(self, pair: Order) -> str:
        if pair[0] == pair[1]:
            raise ValueError("crossing requested on equal labels")
        return f"beta[{label_word(pair)}]"

    def tensor_embed(self, core: str, left: Order, right: Order) -> str:
        if core == "id":
            return "id"
        factors = [self.id_symbol(label) for label in left] + [core] + [self.id_symbol(label) for label in right]
        return " x ".join(factors)

    def parenthesize(self, term: str) -> str:
        if term.startswith("(") and term.endswith(")"):
            return term
        if " o " in term or " x " in term:
            return f"({term})"
        return term

    def compose_terms(self, execution_terms: list[str]) -> str:
        terms = [term for term in execution_terms if term != "id"]
        if not terms:
            return "id"
        return " o ".join(reversed([self.parenthesize(term) for term in terms]))

    def step_word(self, step: Step, current_order: Order, top: bool = False) -> str:
        pos = step.pos - 1
        if step.kind == "cross":
            return self.tensor_embed(
                self.crossing_symbol(current_order[pos : pos + 2]),
                current_order[:pos],
                current_order[pos + 2 :],
            )
        branching_rule = self.branching_rules[step.branching_name]
        width = len(branching_rule.codomain_order if top else branching_rule.domain_order)
        return self.tensor_embed(
            self.branching_symbol(branching_rule, flipped=top),
            current_order[:pos],
            current_order[pos + width :],
        )

    def bottom_word(self, boundary: Order, steps: tuple[Step, ...]) -> str:
        # Print the atom/crossing word for a bottom branch.
        order = boundary
        terms: list[str] = []
        for step in steps:
            terms.append(self.step_word(step, order))
            order = self.apply_step_to_order(order, step)
        return self.compose_terms(terms)

    def top_word(self, boundary: Order, steps: tuple[Step, ...], domain_order: Order) -> str:
        # Print the inverse word for a top branch.
        trace = self.order_trace(boundary, steps)
        order = domain_order
        terms: list[str] = []
        for idx in range(len(steps) - 1, -1, -1):
            step = steps[idx]
            terms.append(self.step_word(step, order, top=True))
            order = trace[idx]
        return self.compose_terms(terms)

    def apply_commutor(self, outputs: list[object], order: list[int], pos: int) -> None:
        # Apply a crossing to adjacent tensor factors.
        pair = tuple(order[pos : pos + 2])
        image = self.sage_commutor(pair, tuple(outputs[pos : pos + 2]))
        outputs[pos : pos + 2] = list(image)
        order[pos : pos + 2] = list(reversed(pair))

    def apply_step_to_order(self, order: Order, step: Step) -> Order:
        # Track how a branch step changes the boundary word.
        pos = step.pos - 1
        if step.kind == "cross":
            return order[:pos] + (order[pos + 1], order[pos]) + order[pos + 2 :]
        branching_rule = self.branching_rules[step.branching_name]
        width = len(branching_rule.domain_order)
        if order[pos : pos + width] != branching_rule.domain_order:
            raise ValueError(f"{step.branching_name}: domain mismatch")
        return order[:pos] + branching_rule.codomain_order + order[pos + width :]

    def execute_steps(self, tensor: Tensor, start_order: Order, steps: tuple[Step, ...]) -> Tensor | None:
        # Apply a whole branch word to a tensor.
        outputs = list(tensor)
        order = list(start_order)
        for step in steps:
            pos = step.pos - 1
            if step.kind == "cross":
                if order[pos] == order[pos + 1]:
                    return None
                self.apply_commutor(outputs, order, pos)
                continue

            branching_rule = self.branching_rules[step.branching_name]
            width = len(branching_rule.domain_order)
            if tuple(order[pos : pos + width]) != branching_rule.domain_order:
                return None
            image = self.apply_branching_rule(branching_rule, tuple(outputs[pos : pos + width]))
            if image is None:
                return None
            outputs[pos : pos + width] = list(image)
            order[pos : pos + width] = list(branching_rule.codomain_order)
        return tuple(outputs)

    def route_suffix(self, order: Order, suffix: Order) -> tuple[tuple[int, ...], Order] | None:
        # Move the rightmost required labels to the right, in the requested order.
        if not suffix:
            return (), order
        if len(suffix) > len(order):
            return None
        if any(order.count(label) < suffix.count(label) for label in set(suffix)):
            return None

        labels = list(order)
        swaps: list[int] = []
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

    def build_branching(self, labels: Order) -> tuple[tuple[BranchNode, ...], ...]:
        # Recursively choose a summand after each new tensor factor is added.
        if not labels:
            identity = BranchNode(zero_weight(self.rank), (), (), lambda tensor: tensor)
            return ((identity,),)

        root_weight = fundamental_weight(self.rank, labels[0])
        root = BranchNode(root_weight, (labels[0],), (), lambda tensor: tensor)
        stages: list[tuple[BranchNode, ...]] = [(root,)]

        for new_label in labels[1:]:
            current_stage: list[BranchNode] = []
            for parent in stages[-1]:
                for choice in self.rules_from(parent.highest_weight, new_label):
                    branching_rule = choice.rule
                    routed = self.route_suffix(parent.order, branching_rule.old_suffix)
                    if routed is None:
                        continue
                    swaps, _ = routed
                    offset = len(parent.order) - len(branching_rule.old_suffix)
                    extension_steps = tuple(Step("cross", pos) for pos in swaps) + (
                        Step("branch", offset + 1, branching_rule.name),
                    )

                    def apply(tensor, parent=parent, new_label=new_label, extension_steps=extension_steps):
                        prefix = parent.apply(tensor[:-1])
                        if prefix is None:
                            return None
                        return self.execute_steps(prefix + (tensor[-1],), parent.order + (new_label,), extension_steps)

                    current_stage.append(
                        BranchNode(
                            choice.child_weight,
                            self.order_trace(parent.order + (new_label,), extension_steps)[-1],
                            parent.steps + extension_steps,
                            apply,
                        )
                    )
            stages.append(tuple(current_stage))
        return tuple(stages)

    def order_trace(self, boundary: Order, steps: tuple[Step, ...]) -> tuple[Order, ...]:
        # Record order changes so the same branch can be traversed backward.
        trace = [boundary]
        order = boundary
        for step in steps:
            order = self.apply_step_to_order(order, step)
            trace.append(order)
        return tuple(trace)

    def apply_top(self, boundary: Order, steps: tuple[Step, ...], domain_order: Order, tensor: Tensor) -> Tensor | None:
        # Apply the inverse of the chosen branch for the codomain word.
        outputs = list(tensor)
        current_order = domain_order
        trace = self.order_trace(boundary, steps)
        for idx in range(len(steps) - 1, -1, -1):
            step = steps[idx]
            before = trace[idx]
            after = trace[idx + 1]
            if current_order != after:
                raise ValueError("top order mismatch")
            pos = step.pos - 1
            if step.kind == "cross":
                order = list(current_order)
                self.apply_commutor(outputs, order, pos)
                current_order = tuple(order)
                continue

            branching_rule = self.branching_rules[step.branching_name]
            width = len(branching_rule.codomain_order)
            image = self.apply_branching_rule(branching_rule, tuple(outputs[pos : pos + width]), inverse=True)
            if image is None:
                return None
            outputs[pos : pos + width] = list(image)
            current_order = before
        if current_order != boundary:
            raise ValueError("top boundary mismatch")
        return tuple(outputs)

    def permutation_swaps(self, domain_order: Order, codomain_order: Order) -> tuple[int, ...]:
        labels = list(domain_order)
        swaps: list[int] = []
        if sorted(labels) != sorted(codomain_order):
            raise ValueError("orders have different multisets")
        # Fill the codomain from right to left.  On three distinct strands this
        # chooses the left-starting Reidemeister-3 word.
        for idx in range(len(codomain_order) - 1, -1, -1):
            desired_label = codomain_order[idx]
            if labels[idx] == desired_label:
                continue
            swap_idx = next(pos for pos in range(idx - 1, -1, -1) if labels[pos] == desired_label)
            for pos in range(swap_idx, idx):
                if labels[pos] == labels[pos + 1]:
                    raise ValueError("equal-label crossings are not used")
                labels[pos], labels[pos + 1] = labels[pos + 1], labels[pos]
                swaps.append(pos + 1)
        return tuple(swaps)

    def commute_to_order(self, tensor: Tensor, domain_order: Order, codomain_order: Order) -> Tensor:
        # Apply the middle permutation to a tensor.
        outputs = list(tensor)
        order = list(domain_order)
        for swap_pos in self.permutation_swaps(domain_order, codomain_order):
            self.apply_commutor(outputs, order, swap_pos - 1)
        return tuple(outputs)

    def permutation_word(self, domain_order: Order, codomain_order: Order) -> str:
        # Print the middle permutation word.
        order = list(domain_order)
        terms: list[str] = []
        for swap_pos in self.permutation_swaps(domain_order, codomain_order):
            pos = swap_pos - 1
            current_order = tuple(order)
            terms.append(
                self.tensor_embed(
                    self.crossing_symbol(current_order[pos : pos + 2]),
                    current_order[:pos],
                    current_order[pos + 2 :],
                )
            )
            order[pos], order[pos + 1] = order[pos + 1], order[pos]
        return self.compose_terms(terms)

    def bpt_basis(self, domain: Order, codomain: Order) -> tuple[Diagram, ...]:
        # Pair branches with the same highest weight; this gives the Hom basis.
        bottom_nodes = self.build_branching(domain)[-1]
        top_nodes = self.build_branching(codomain)[-1]
        return tuple(
            Diagram(self, domain, codomain, bottom, top)
            for bottom in bottom_nodes
            for top in top_nodes
            if bottom.highest_weight == top.highest_weight
        )

    def decomposition_counts(self, word: Order) -> Counter[Weight]:
        if not word:
            return Counter({zero_weight(self.rank): 1})
        return Counter(self.sage_weight(element) for element in self.sage_parent(word).highest_weight_vectors())

    def expected_hom_dimension(self, domain: Order, codomain: Order) -> int:
        domain_counts = self.decomposition_counts(domain)
        codomain_counts = self.decomposition_counts(codomain)
        return sum(domain_counts[weight] * codomain_counts[weight] for weight in domain_counts.keys() & codomain_counts.keys())

    def independence_rank(self, domain: Order, codomain: Order, diagrams: tuple[Diagram, ...]) -> int:
        # It is enough to test these crystal maps on highest-weight tensors.
        domain_highest = self.highest_tensors(domain)
        codomain_highest = self.highest_tensors(codomain)
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
    parser = argparse.ArgumentParser(description="List a classical bottom-permutation-top basis.")
    parser.add_argument("cartan_type", help="A3, B4, C3, D5, ...")
    parser.add_argument("domain")
    parser.add_argument("codomain")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--check-independence", action="store_true")
    args = parser.parse_args()

    try:
        context = ClassicalBPT(parse_cartan_type(args.cartan_type))
        domain = parse_word(args.domain, context.rank)
        codomain = parse_word(args.codomain, context.rank)
    except ValueError as error:
        parser.error(str(error))

    basis = context.bpt_basis(domain, codomain)
    expected = context.expected_hom_dimension(domain, codomain)
    print(f"{args.cartan_type.upper()} {fmt_order(domain)} -> {fmt_order(codomain)}")
    print(f"basis diagrams: {len(basis)}")
    print(f"expected Hom dimension: {expected}")
    if args.check_independence:
        rank = context.independence_rank(domain, codomain, basis)
        print(f"highest-weight rank: {rank}")
        print(f"independent: {rank == len(basis)}")
    if args.count_only:
        return
    for idx, diagram in enumerate(basis, start=1):
        print(f"{idx}. wt={fmt_weight(diagram.weight)} {diagram.word()}")


if __name__ == "__main__":
    main()
