Run the Python files with Sage's Python, for example

  sage -python g2_bpt.py 112 21

Words are written as strings of fundamental labels: 112 means (1,1,2), 21
means (2,1), and 0 means the empty word.  For labels with more than one digit,
use commas or spaces, for example 1,2,10.


g2_bpt.py and classical_bpt.py
------------------------------

Lists the bottom-perm-top basis for a G2/classical types A,B,C,D hom-space.

sage -python g2_bpt.py 112 21
sage -python classical_bpt.py C3 1,2,1 3 --count-only

Typical output starts with

  (1,1,2) -> (2,1)
  basis diagrams: 7
  expected Hom dimension: 7

and then lists the diagrams.  To print only the size:

  sage -python g2_bpt.py 112 21 --count-only

An optional independence check is available:

  sage -python g2_bpt.py 112 21 --check-independence





g2_atom_relations.py and classical_atom_relations.py
----------------------------------------------------

Enumerates G2/classical atom relations and expands them in the BPT basis.

  sage -python g2_atom_relations.py --family HI1 --limit 5
  sage -python classical_atom_relations.py A3 --family all --count-only

Families are HI1, HI2, HI3, OR1, OR2, OR3 (for classical types also OR4).  A displayed relation looks like

  1. HI1 [11->0] o bar([11->0]) : (0) -> (0)
     = id

Here x denotes tensor product and o denotes function composition: in f o g,
apply g first, then f.

Useful flags:

  --family HI2
  --family all
  --count-only
  --limit 10   
  --show-zero (show the relations whose RHS is 0 as well)





g2_relations.html
-----------------

Rendered G2 relations.

