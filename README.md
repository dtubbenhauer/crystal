# Code and Erratum for *Presentations for categories of crystals*

This repository contains Sage/Python code accompanying the paper

> David He and Daniel Tubbenhauer, *Presentations for categories of crystals*.

The code enumerates the bottom-permutation-top bases and the local atom relations used in the paper.  It covers the classical types `A`, `B`, `C`, `D`, and type `G2`.  The repository also contains a rendered HTML list of the `G2` relations, since that list is too long to print comfortably in the paper.

## Contact

If you find any errors in the paper **please email me**:

[dtubbenhauer@gmail.com](mailto:dtubbenhauer@gmail.com?subject=[GitHub]%web-reps)

Same goes for any errors related to this page.

## Requirements

The scripts should be run with SageMath's Python:

```bash
sage -python <script.py> ...
```

The code uses Sage's crystals, Cartan types, matrices and rational arithmetic.  Running the scripts with ordinary Python will not work unless Sage is available in that Python environment.

## Files

| file | purpose |
|---|---|
| `classical_bpt.py` | enumerates bottom-permutation-top bases for hom-spaces in classical types `A`, `B`, `C`, `D` |
| `g2_bpt.py` | enumerates bottom-permutation-top bases for hom-spaces in type `G2` |
| `classical_atom_relations.py` | enumerates local atom relations in classical types and expands them in the BPT basis |
| `g2_atom_relations.py` | enumerates local atom relations in type `G2` and expands them in the BPT basis |
| `g2_relations.html` | rendered list of the `G2` atom relations |

## Word notation

Boundary words are written as strings of fundamental labels.  For example,

```text
112
```

means the tensor word `(1,1,2)`, and

```text
21
```

means `(2,1)`.  The symbol `0` denotes the empty word.  For labels with more than one digit, use commas or spaces, for example

```text
1,2,10
```

or

```text
1 2 10
```

## Bottom-permutation-top bases

For type `G2`:

```bash
sage -python g2_bpt.py 112 21
```

For a classical type:

```bash
sage -python classical_bpt.py C3 1,2,1 3
```

Typical output begins with something like

```text
(1,1,2) -> (2,1)
basis diagrams: 7
expected Hom dimension: 7
```

To print only the size of the basis, use `--count-only`:

```bash
sage -python g2_bpt.py 112 21 --count-only
sage -python classical_bpt.py C3 1,2,1 3 --count-only
```

An optional independence check is available:

```bash
sage -python g2_bpt.py 112 21 --check-independence
sage -python classical_bpt.py C3 1,2,1 3 --check-independence
```

## Atom relations

For type `G2`:

```bash
sage -python g2_atom_relations.py --family HI1 --limit 5
```

For a classical type:

```bash
sage -python classical_atom_relations.py A3 --family all --count-only
```

Available relation families are:

```text
HI1, HI2, HI3, OR1, OR2, OR3
```

For classical types, `OR4` is also available.

A displayed relation looks like

```text
1. HI1 [11->0] o bar([11->0]) : (0) -> (0)
   = id
```

Here `x` denotes tensor product and `o` denotes function composition: in `f o g`, apply `g` first and then `f`.

Useful flags:

```bash
--family HI2       # choose one relation family
--family all       # enumerate all implemented families
--count-only       # only print the number of relations
--limit 10         # print only the first 10 relations
--show-zero        # also display relations whose right-hand side is zero
```

For classical types there is also

```bash
--max-atom-len N
--max-middle-len N
```

which restrict the enumeration to smaller local configurations.

## Rendered `G2` relations

Open

```text
g2_relations.html
```

in a browser to view the rendered `G2` relations.  The page uses MathJax from a CDN, so an internet connection is needed for the formulas to render unless MathJax is served locally.

## Suggested citation

If you use this code, please cite the accompanying paper:

```bibtex
@misc{HeTubbenhauerCrystalPresentations,
  author = {He, David and Tubbenhauer, Daniel},
  title = {Presentations for categories of crystals},
  year = {2026}
}
```

## Erratum

Empty so far.
