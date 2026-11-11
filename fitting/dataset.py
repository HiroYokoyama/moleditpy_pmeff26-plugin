"""Reference molecule sets for the PEFF26 parameter fit.

Three sets, all neutral closed-shell molecules over H, B, C, N, O, F, Si, P,
S, Cl, Br and I:

* ``TRAIN``    — small molecules covering the bond, angle and out-of-plane
  types the fit parameterizes (equilibrium geometry + Hessian).
* ``VALIDATE`` — larger, held-out molecules. They are never used by the fit;
  the reported accuracy comes from these.
* ``SCANS``    — relaxed torsion scans: ``(SMILES, (i, j, k, l))`` with the
  dihedral given as heavy-atom SMILES indices (RDKit order, before AddHs) or
  explicit-H indices after AddHs.

Reference level: wB97X-D/def2-TZVP (GAMESS).
"""

from __future__ import annotations

ALLOWED_ELEMENTS = frozenset(
    {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
)

TRAIN = """
# --- hydrides and diatomics
[HH] C N O F [SiH4] P S Cl Br I B
N#N FF ClCl BrBr II FCl ClBr ICl IBr [C-]#[O+]
# --- alkanes and rings
CC CCC CCCC CC(C)C CC(C)(C)C CCCCC CCC(C)C
C1CC1 C1CCC1 C1CCCC1 C1CCCCC1 CC1CC1 C1CC12CC2 C1CC2CCC1C2 CC1CCCC1
# --- alkenes, allenes, alkynes
C=C CC=C C=CC=C C/C=C/C C/C=C\\C CC(C)=C CC(C)=C(C)C C=C=C C=CC=CC=C
C1=CC1 C1=CCC1 C1=CCCC1 C1=CCCCC1 C1=CC=CC1 C1=CCC=CC1 C=C1CCC1
C#C CC#C CC#CC C#CC#C C=CC#C CCC#C
# --- arenes
c1ccccc1 Cc1ccccc1 Cc1ccccc1C C=Cc1ccccc1 C#Cc1ccccc1 c1ccc2ccccc2c1
# --- heteroaromatics
c1ccncc1 c1ccnnc1 c1cncnc1 c1cnccn1 c1ncncn1 c1cc[nH]c1 c1ccoc1 c1ccsc1
c1c[nH]cn1 c1cn[nH]c1 c1cocn1 c1cscn1 c1cnoc1 c1cnsc1 c1nc[nH]n1 c1nnn[nH]1
Cn1cccc1 Cc1ccco1 Cc1cccs1 Cc1ccccn1 c1ccc2[nH]ccc2c1 c1ccc2occc2c1
c1ccc2sccc2c1 c1ncc2[nH]cnc2n1 c1cc[n+]([O-])cc1
# --- alcohols, ethers, peroxides
CO CCO CC(C)O CC(C)(C)O CCCO COC CCOC C1CO1 C1COC1 C1CCOC1 C1CCOCC1
C1COCCO1 OCCO COCOC COCCOC OO COO COOC Oc1ccccc1 COc1ccccc1 C=CO C=COC
OC1CC1 OCC=C OCC#C
# --- aldehydes, ketones, cumulated C=O
C=O CC=O CCC=O CC(C)=O CCC(C)=O O=CC=O C=CC=O CC(=O)C=C O=C=O C=C=O
O=C1CC1 O=C1CCC1 O=C1CCCC1 O=C1CCCCC1 O=Cc1ccccc1 CC(=O)c1ccccc1
O=C1C=CC(=O)C=C1 CC(=O)CC(C)=O OCC=O CC(=O)C(C)=O
# --- acids, esters, anhydrides, carbonates
OC=O CC(=O)O COC=O CC(=O)OC CCOC=O O=C1CCO1 O=C1CCCO1 OC(=O)O COC(=O)OC
CC(=O)OC(C)=O OC(=O)c1ccccc1 OC(=O)C=C OC(=O)C(=O)O O=C1OCCO1 OCC(=O)O
CC(O)C(=O)O
# --- amines, hydrazines, hydroxylamines
CN CNC CN(C)C CCN CC(C)N C1CN1 C1CNC1 C1CCNC1 C1CCNCC1 NCCN Nc1ccccc1
CNc1ccccc1 NN CNN CN(C)N NO CON NCCO C1COCCN1 C1CNCCN1 C=CN NC1CC1
# --- imines, oximes, nitriles, isocyanides, azo, cumulenes
C=N CC=N CN=C CC(C)=NC C=NO CC(C)=NO C=NN C#N CC#N CCC#N C=CC#N N#CC#N
N#Cc1ccccc1 [C-]#[N+]C N=C=O CN=C=O N=C=N C=[N+]=[N-] N=[N+]=[N-]
CN=[N+]=[N-] N=N C/N=N/C NCC#N OCC#N N#CN
# --- nitro, nitroso, nitrite, nitrate, N-oxides
C[N+](=O)[O-] [O-][N+](=O)c1ccccc1 CN=O ON=O CON=O O[N+](=O)[O-]
CO[N+](=O)[O-] N#[N+][O-] C[N+](C)(C)[O-] CC[N+](=O)[O-]
# --- amides, ureas, carbamates, amidines, lactams
NC=O CNC=O CN(C)C=O CC(N)=O CC(=O)NC NC(N)=O CNC(N)=O NC(=O)O COC(N)=O
O=C1CCCN1 O=C1CCN1 O=C1CCC(=O)N1 NC(=O)c1ccccc1 NC(N)=N CC(N)=N
O=c1cccc[nH]1 O=c1cc[nH]c(=O)[nH]1 CC(=O)N(C)C NCC(=O)O CC(N)C(=O)O
NC(=O)C=C
# --- fluorine
CF FCF FC(F)F FC(F)(F)F CCF FCCF CC(F)F CC(F)(F)F C=CF FC=C(F)F
FC(F)=C(F)F Fc1ccccc1 Fc1ccc(F)cc1 O=C(F)F CC(F)=O FC#C OC(=O)C(F)(F)F
FC(F)(F)C=O OCC(F)(F)F NF OF FOF FC(F)(F)C#N
# --- chlorine
CCl ClCCl ClC(Cl)Cl ClC(Cl)(Cl)Cl CCCl ClCCCl CC(Cl)Cl C=CCl Cl/C=C/Cl
ClC(Cl)=C(Cl)Cl Clc1ccccc1 CC(Cl)=O O=C(Cl)Cl ClC#C OCl NCl OC(=O)CCl
Clc1ccncc1 FC(F)(F)Cl CC(C)(C)Cl ClC#N
# --- bromine
CBr BrCBr BrC(Br)Br CCBr BrCCBr C=CBr Brc1ccccc1 CC(Br)=O OBr BrC#C
Brc1cccs1 FC(F)(F)Br CC(C)Br BrC#N ClCBr
# --- iodine
CI ICI IC(I)I CCI C=CI Ic1ccccc1 FC(F)(F)I CC(C)I IC#C IC#N OI ICCI
# --- sulfur
CS CSC CCS CCSC SS CSSC C1CS1 C1CCSC1 C=S CC(C)=S S=C=S O=C=S CSC=O
CC(=O)S CC(N)=S NC(N)=S N#CS CN=C=S CSC#N Sc1ccccc1 CSc1ccccc1 C=CS
CS(C)=O CS(C)(=O)=O O=S=O O=S(=O)=O CS(=O)(=O)O CS(N)(=O)=O OS(O)(=O)=O
COS(=O)(=O)OC O=S1(=O)CCCC1 FS(F)(F)(F)(F)F CS(=O)(=O)F CS(=O)(=O)Cl
ClSCl ClS(Cl)=O O=S(=O)(Cl)Cl FSF OS(O)=O COS(=O)OC SCCS CS(=O)O NS
# --- phosphorus
CP CPC CP(C)C PP C#P CP(C)(C)=O OP(O)(O)=O COP(=O)(OC)OC COP(OC)OC
CP(C)(C)=S O=P(F)(F)F FP(F)(F)(F)F CP(=O)(O)O NP Pc1ccccc1 FP(F)F
ClP(Cl)Cl O=P(Cl)(Cl)Cl BrP(Br)Br OP(O)O CP(C)Cl C=CP OP CCP
# --- silicon
C[SiH3] C[SiH2]C C[Si](C)(C)C [SiH3][SiH3] O[SiH3] [SiH3]O[SiH3]
CO[SiH3] C[Si](C)(C)O N[SiH3] [SiH3]N[SiH3] S[SiH3] [SiH3]c1ccccc1
C[Si](C)(C)Cl [SiH3]C#C C=C[SiH3] F[Si](F)(F)F Cl[Si](Cl)(Cl)Cl
C1C[SiH2]C1 [SiH3]F [SiH3]Cl [SiH3]Br [SiH3]I CO[Si](OC)(OC)OC CC[SiH3]
[SiH3]P [SiH3]C#N
# --- boron
CB CB(C)C OB(O)O COB(OC)OC FB(F)F ClB(Cl)Cl BrB(Br)Br NB [NH3+][BH3-]
B1NBNBN1 CB(O)O OB(O)c1ccccc1 CB(F)F CN(C)B NB(N)N SB OB C=CB
"""

VALIDATE = """
# --- hydrocarbons
CCCCCC CC=C(C)C CC1CCCCC1 C1CC2C=CC1C2 C1C2CC3CC1CC(C2)C3 CCc1ccccc1
C(#Cc1ccccc1)C Cc1ccc(C)cc1 C=CCC=C CC(C)C#C
# --- oxygen
CCOCC COC(C)(C)C OC1CCCCC1 C1COCO1 CCOC(C)=O COC(=O)c1ccccc1
CC(C)(C)C(C)=O C1OC1c1ccccc1 Cc1ccc(C)o1 O=Cc1ccco1 CC(=O)Oc1ccccc1
OCC(O)CO Oc1ccc(O)cc1 CCC(=O)O
# --- nitrogen
CCN(CC)CC CC(=O)Nc1ccc(O)cc1 CN1CCCC1=O O=C1CCCCCN1 NC(=O)c1cccnc1
Nc1cc[nH]c(=O)n1 Cc1c[nH]c(=O)[nH]c1=O Nc1ncccn1 c1ccc2[nH]cnc2c1
c1ccc2ncccc2c1 OC(=O)C1CCCN1 NC(CO)C(=O)O CCOC(N)=O CN(C)c1ccccc1
Cc1ccc(cc1)[N+](=O)[O-] N#Cc1ccncc1 CC(C)=NN
# --- halogens
Nc1ccc(F)cc1 Oc1ccc(Cl)cc1 Brc1ccccn1 Cc1cccc(I)c1 OCCCl OCCF CC(=O)CBr
CCCI CC(Cl)(Cl)Cl FC(F)(F)C(Cl)Br Fc1c(F)c(F)c(F)c(F)c1F ClC=C(Cl)Cl
FC(F)(F)c1ccccc1 Clc1ccc(Cl)cc1
# --- sulfur, phosphorus, silicon, boron
C1CCSCC1 CCSSCC NS(=O)(=O)c1ccccc1 COS(C)(=O)=O O=Cc1cccs1 Nc1nccs1
c1ccc2scnc2c1 NC(CS)C(=O)O Cc1ccc(S)cc1 CCS(=O)CC COP(C)(=O)OC
CCP(CC)CC CP(=O)(c1ccccc1)C C[Si](C)(C)C#C C[Si](C)(C)O[Si](C)(C)C
C[Si](C)(C)c1ccccc1 CCO[SiH3] COB(C)OC CB1OCCO1 CCB(CC)CC
"""

# Relaxed scans: SMILES and the scanned dihedral as indices into the molecule
# *after* AddHs. ``None`` for an index means "first hydrogen on that atom"
# (resolved by make_pack), which keeps methyl/hydroxyl rotors readable.
# Each entry: (label, smiles, (i, j, k, l)) where i/l may be "H" markers.
SCANS = [
    # sp3-sp3
    ("ethane", "CC", ("H", 0, 1, "H")),
    ("propane", "CCC", ("H", 0, 1, 2)),
    ("butane", "CCCC", (0, 1, 2, 3)),
    ("methanol", "CO", ("H", 0, 1, "H")),
    ("ethanol_cc", "CCO", ("H", 0, 1, 2)),
    ("ethanol_co", "CCO", (0, 1, 2, "H")),
    ("dimethyl_ether", "COC", ("H", 0, 1, 2)),
    ("ethyl_methyl_ether", "CCOC", (0, 1, 2, 3)),
    ("methylamine", "CN", ("H", 0, 1, "H")),
    ("ethylamine_cn", "CCN", (0, 1, 2, "H")),
    ("dimethylamine", "CNC", ("H", 0, 1, 2)),
    ("methanethiol", "CS", ("H", 0, 1, "H")),
    ("ethanethiol_cs", "CCS", (0, 1, 2, "H")),
    ("ethyl_methyl_sulfide", "CCSC", (0, 1, 2, 3)),
    ("hydrogen_peroxide", "OO", ("H", 0, 1, "H")),
    ("dimethyl_disulfide", "CSSC", (0, 1, 2, 3)),
    ("hydrazine", "NN", ("H", 0, 1, "H")),
    ("hydroxylamine", "NO", ("H", 0, 1, "H")),
    ("difluoroethane", "FCCF", (0, 1, 2, 3)),
    ("dichloroethane", "ClCCCl", (0, 1, 2, 3)),
    ("fluoroethane", "CCF", ("H", 0, 1, 2)),
    ("chloroethane", "CCCl", ("H", 0, 1, 2)),
    ("bromoethane", "CCBr", ("H", 0, 1, 2)),
    ("methylsilane", "C[SiH3]", ("H", 0, 1, "H")),
    ("methylphosphine", "CP", ("H", 0, 1, "H")),
    ("ethylene_glycol", "OCCO", (0, 1, 2, 3)),
    ("ethylenediamine", "NCCN", (0, 1, 2, 3)),
    ("propanol_cc", "CCCO", (0, 1, 2, 3)),
    ("disilane", "[SiH3][SiH3]", ("H", 0, 1, "H")),
    ("methoxysilane", "CO[SiH3]", (0, 1, 2, "H")),
    # sp2-sp3
    ("propene", "CC=C", ("H", 0, 1, 2)),
    ("toluene", "Cc1ccccc1", ("H", 0, 1, 2)),
    ("acetaldehyde", "CC=O", ("H", 0, 1, 2)),
    ("acetone", "CC(C)=O", ("H", 0, 1, 3)),
    ("acetic_acid_cc", "CC(=O)O", ("H", 0, 1, 2)),
    ("butene", "CCC=C", (0, 1, 2, 3)),
    ("propanal", "CCC=O", (0, 1, 2, 3)),
    ("ethylbenzene", "CCc1ccccc1", (0, 1, 2, 3)),
    ("methyl_formate_och3", "COC=O", ("H", 0, 1, 2)),
    ("anisole", "COc1ccccc1", (0, 1, 2, 3)),
    ("n_methylformamide_nc", "CNC=O", ("H", 0, 1, 2)),
    ("nitromethane", "C[N+](=O)[O-]", ("H", 0, 1, 2)),
    ("methyl_vinyl_ether", "COC=C", (0, 1, 2, 3)),
    ("thioanisole", "CSc1ccccc1", (0, 1, 2, 3)),
    ("dmso", "CS(C)=O", ("H", 0, 1, 3)),
    # sp2-sp2
    ("butadiene", "C=CC=C", (0, 1, 2, 3)),
    ("acrolein", "C=CC=O", (0, 1, 2, 3)),
    ("glyoxal", "O=CC=O", (0, 1, 2, 3)),
    ("styrene", "C=Cc1ccccc1", (0, 1, 2, 3)),
    ("benzaldehyde", "O=Cc1ccccc1", (0, 1, 2, 3)),
    ("formic_acid", "O=CO", (0, 1, 2, "H")),
    ("methyl_formate", "O=COC", (0, 1, 2, 3)),
    ("formamide", "O=CN", (0, 1, 2, "H")),
    ("n_methylformamide", "O=CNC", (0, 1, 2, 3)),
    ("phenol", "Oc1ccccc1", ("H", 0, 1, 2)),
    ("vinyl_alcohol", "C=CO", (0, 1, 2, "H")),
    ("aniline", "Nc1ccccc1", ("H", 0, 1, 2)),
    ("thiophenol", "Sc1ccccc1", ("H", 0, 1, 2)),
    ("nitrobenzene", "[O-][N+](=O)c1ccccc1", (0, 1, 3, 4)),
    ("acrylic_acid", "C=CC(=O)O", (0, 1, 2, 3)),
    ("thioformic_acid", "O=CS", (0, 1, 2, "H")),
    ("vinylborane", "C=CB", (0, 1, 2, "H")),
    ("biphenyl", "c1ccccc1-c1ccccc1", (0, 5, 6, 7)),
]


def _parse(block: str) -> list[str]:
    out: list[str] = []
    for line in block.splitlines():
        line = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
        out.extend(line.split())
    return out


def train_smiles() -> list[str]:
    return _parse(TRAIN)


def validate_smiles() -> list[str]:
    return _parse(VALIDATE)
