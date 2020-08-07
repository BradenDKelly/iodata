# IODATA is an input and output module for quantum chemistry.
# Copyright (C) 2011-2020 The IODATA Development Team
#
# This file is part of IODATA.
#
# IODATA is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# IODATA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --

from typing import Tuple, List, TextIO, Iterator

import numpy as np

from ..basis import HORTON2_CONVENTIONS, MolecularBasis, Shell
from ..docstrings import document_load_one, document_dump_one
from ..iodata import IOData
from ..orbitals import MolecularOrbitals
from ..periodic import num2sym
from ..utils import LineIterator

__all__ = []

PATTERNS = ['*.mwfn']

# From the MWFN chemrxiv paper
# https://chemrxiv.org/articles/Mwfn_A_Strict_Concise_and_Extensible_Format
# _for_Electronic_Wavefunction_Storage_and_Exchange/11872524
# For cartesian shells
# S shell: S
# P shell: X, Y, Z
# D shell: XX, YY, ZZ, XY, XZ, YZ
# F shell: XXX, YYY, ZZZ, XYY, XXY, XXZ, XZZ, YZZ, YYZ, XYZ
# G shell: ZZZZ, YZZZ, YYZZ, YYYZ, YYYY, XZZZ, XYZZ, XYYZ, XYYY, XXZZ, XXYZ, XXYY, XXXZ, XXXY, XXXX
# H shell: ZZZZZ, YZZZZ, YYZZZ, YYYZZ, YYYYZ, YYYYY, XZZZZ, XYZZZ, XYYZZ, XYYYZ, XYYYY, XXZZZ, XXYZZ,
#          XXYYZ, XXYYY, XXXZZ, XXXYZ, XXXYY, XXXXZ, XXXXY, XXXXX
# For pure shells, the order is
# D shell: D 0, D+1, D-1, D+2, D-2
# F shell: F 0, F+1, F-1, F+2, F-2, F+3, F-3
# G shell: G 0, G+1, G-1, G+2, G-2, G+3, G-3, G+4, G-4

CONVENTIONS = {
    (9, 'p'): HORTON2_CONVENTIONS[(9, 'p')],
    (8, 'p'): HORTON2_CONVENTIONS[(8, 'p')],
    (7, 'p'): HORTON2_CONVENTIONS[(7, 'p')],
    (6, 'p'): HORTON2_CONVENTIONS[(6, 'p')],
    (5, 'p'): HORTON2_CONVENTIONS[(5, 'p')],
    (4, 'p'): HORTON2_CONVENTIONS[(4, 'p')],
    (3, 'p'): HORTON2_CONVENTIONS[(3, 'p')],
    (2, 'p'): HORTON2_CONVENTIONS[(2, 'p')],
    (0, 'c'): ['1'],
    (1, 'c'): ['x', 'y', 'z'],
    (2, 'c'): ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
    (3, 'c'): ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz'],
    (4, 'c'): HORTON2_CONVENTIONS[(4, 'c')][::-1],
    (5, 'c'): HORTON2_CONVENTIONS[(5, 'c')][::-1],
    (6, 'c'): HORTON2_CONVENTIONS[(6, 'c')][::-1],
    (7, 'c'): HORTON2_CONVENTIONS[(7, 'c')][::-1],
    (8, 'c'): HORTON2_CONVENTIONS[(8, 'c')][::-1],
    (9, 'c'): HORTON2_CONVENTIONS[(9, 'c')][::-1],
}


def _load_helper_num(lit: LineIterator) -> List[int]:
    """Read number of orbitals, primitives and atoms."""
    line = next(lit)
    if not line.startswith('G'):
        lit.error("Expecting line to start with G.")
    return [int(i) for i in line.split() if i.isdigit()]


def _load_helper_opener(lit: LineIterator) -> float:
    """Read initial variables."""
    opener_keywords = ["Wfntype", "Charge", "Naelec", "Nbelec", "E_tot", "VT_ratio", "Ncenter"]
    max_count = len(opener_keywords)
    count = 0
    d = {}
    # line = next(lit).lower()
    while count < max_count:
        line = next(lit)
        for name in opener_keywords:
            if name in line:
                d[name] = line.split('=')[1].strip()
                count += 1
    print(float(d['VT_ratio']))
    return int(d['Wfntype']), float(d['Charge']), float(d['Naelec']), float(d['Nbelec']), \
           float(d['E_tot']), float(d['VT_ratio']), int(d['Ncenter'])


def _load_helper_basis(lit: LineIterator) -> Tuple[int, int, int, int, int]:
    """Read initial variables."""
    # Nprims must be last or else it gets read in with Nprimshell
    basis_keywords = ["Nbasis", "Nindbasis", "Nshell", "Nprimshell", "Nprims", ]
    max_count = len(basis_keywords)
    count = 0
    d = {}
    line = next(lit)
    while count < max_count:
        line = next(lit)
        for name in basis_keywords:
            if name in line:
                d[name] = int(line.split('=')[1].strip())
                count += 1
                break
    return d['Nbasis'], d['Nindbasis'], d['Nprims'], d['Nshell'], d['Nprimshell']


def _load_helper_atoms(lit: LineIterator, num_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read the coordinates of the atoms."""
    atnums = np.empty(num_atoms, int)
    atcorenums = np.empty(num_atoms, float)
    atcoords = np.empty((num_atoms, 3), float)
    line = next(lit)
    while '$Centers' not in line and line is not None:
        line = next(lit)

    for atom in range(num_atoms):
        line = next(lit)
        atnums[atom] = int(line.split()[2].strip())
        atcorenums[atom] = float(line.split()[3].strip())
        # extract atomic coordinates
        coords = line.split()
        atcoords[atom, :] = [coords[4], coords[5], coords[6]]

    return atnums, atcorenums, atcoords


def _load_helper_shells(lit: LineIterator, nshell: int, starts: str)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one section of MO information."""
    line = next(lit)
    while starts[0] not in line and line is not None:
        line = next(lit)
    assert line.startswith('$' + starts[0])
    shell_types = _load_helper_section(lit, nshell, ' ', 0, int)
    line = next(lit)
    assert line.startswith('$' + starts[1])
    centers = _load_helper_section(lit, nshell, ' ', 0, int)
    line = next(lit)
    assert line.startswith('$' + starts[2])
    degrees = _load_helper_section(lit, nshell, ' ', 0, int)
    return shell_types, centers, degrees


def _load_helper_prims(lit: LineIterator, nprimshell: int) -> np.ndarray:
    """Read SHELL CENTER, SHELL TYPE, and SHELL CONTRACTION DEGREES sections."""
    line = next(lit)
    # concatenate list of arrays into a single array of length nshell
    array = _load_helper_section(lit, nprimshell, '', 0, float)
    assert len(array) == nprimshell
    return array


def _load_helper_section(lit: LineIterator, nprim: int, start: str, skip: int,
                         dtype: np.dtype) -> np.ndarray:
    """Read SHELL CENTER, SHELL TYPE, and SHELL CONTRACTION DEGREES sections."""
    section = []
    while len(section) < nprim:
        line = next(lit)
        assert line.startswith(start)
        words = line.split()
        section.extend(words[skip:])
    assert len(section) == nprim
    return np.array([word for word in section]).astype(dtype)


def _load_helper_mo(lit: LineIterator, nbasis: int) -> Tuple[int, float, float,
                                                             np.ndarray, int, str]:
    """Read one section of MO information."""
    line = next(lit)
    while 'Index' not in line:
        line = next(lit)

    assert line.startswith('Index')
    number = int(line.split()[1])
    mo_type = int(next(lit).split()[1])
    energy = float(next(lit).split()[1])
    occ = float(next(lit).split()[1])
    sym = str(next(lit).split()[1])
    next_line = next(lit)
    coeffs = _load_helper_section(lit, nbasis, '', 0, float)
    return number, occ, energy, coeffs, mo_type, sym


def load_mwfn_low(lit: LineIterator) -> dict:
    """Load data from a MWFN file into arrays.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Note:
    ---------
    mwfn is a fortran program which loads *.mwfn by locating the line with the keyword,
     `backspace`, then reading. Despite this flexibility, it is stated by the authors that
     the order of section, and indeed, entries in general, must be fixed. With this in mind
     the input utilized some hardcoding since order should be fixed.

     mwfn ignores lines beginning with `#`.

    """
    # read sections of mwfn file
    title = next(lit).strip()  # Generated by Multiwfn // not sure if actual titles are output

    wfntype, charge, nelec_a, nelec_b, energy, vt_ratio, num_atoms = _load_helper_opener(lit)
    # coordinates are in Angstrom in MWFN
    atnums, atcorenums, atcoords = _load_helper_atoms(lit, num_atoms)
    nbasis, nindbasis, nprim, nshell, nprimshell = _load_helper_basis(lit)
    keywords = ["Shell types", "Shell centers", "Shell contraction"]
    shell_types, shell_centers, prim_per_shell = _load_helper_shells(lit, nshell, keywords)
    # HORTON indices start at 0 because Pythons do.
    shell_centers -= 1
    assert wfntype < 5
    assert num_atoms > 0
    assert min(atnums) >= 0
    assert len(shell_types) == nshell
    assert len(shell_centers) == nshell
    assert len(prim_per_shell) == nshell
    exponent = _load_helper_prims(lit, nprimshell)
    coeffs = _load_helper_prims(lit, nprimshell)
    # number of MO's should equal number of independent basis functions. MWFN inc. virtual orbitals.
    num_coeffs = nindbasis
    if wfntype in [0, 2, 3]:
        # restricted wave function
        num_mo = nindbasis
    elif wfntype in [1, 4]:
        # unrestricted wavefunction
        num_mo = 2 * nindbasis

    mo_numbers = np.empty(num_mo, int)
    mo_type = np.empty(num_mo, int)
    mo_occs = np.empty(num_mo, float)
    mo_sym = np.empty(num_mo, str)
    mo_energies = np.empty(num_mo, float)
    mo_coeffs = np.empty([num_coeffs, num_mo], float)

    for mo in range(num_mo):
        mo_numbers[mo], mo_occs[mo], mo_energies[mo], mo_coeffs[:, mo], \
        mo_type[mo], mo_sym[mo] = _load_helper_mo(lit, num_coeffs)

    # TODO add density matrix and overlap

    return {'title': title, 'energy': energy, 'wfntype': wfntype,
            'nelec_a': nelec_a, 'nelec_b': nelec_b, 'charge': charge,
            'atnums': atnums, 'atcoords': atcoords, 'atcorenums': atcorenums,
            'nbasis': nbasis, 'nindbasis': nindbasis, 'nprims': nprim, 'nshells': nshell,
            'nprimshells': nprimshell, 'full_virial_ratio': vt_ratio,
            'shell_centers': shell_centers, 'shell_types': shell_types, 'prim_per_shell': prim_per_shell,
            'exponents': exponent, 'coeffs': coeffs,
            'mo_numbers': mo_numbers, 'mo_occs': mo_occs, 'mo_energies': mo_energies,
            'mo_coeffs': mo_coeffs, 'mo_type': mo_type, 'mo_sym': mo_sym}


def build_obasis(shell_map: np.ndarray, shell_types: np.ndarray,
                 exponents: np.ndarray, prim_per_shell: np.ndarray,
                 coeffs: np.ndarray,
                 lit: LineIterator) -> Tuple[MolecularBasis, np.ndarray]:
    """Based on the fchk modules basis building.

    Parameters
    -------------
    lit
        The line iterator to read the data from.
    shell_map:  np.ndarray (integer)
        Index of what atom the shell is centered on. The mwfn file refers to this section
        as `Shell centers`. Mwfn indices start at 1, this has been modified and starts
        at 0 here. For water (O, H, H) with 6-31G, this would be an array like
        [0, 0, 0, 0, 0, 1, 1, 2, 2]. , `O` in 6-31G has 5 shells and`H` has two shells.
    shell_types: np.ndarray (integer)
        Angular momentum of the shell. Indices start at 0 for 's' orbital, 1 for 'p' etc.
        For 6-31G for a heavy atom this would be [0, 0, 1, 0, 1] corresponding
         to [1s, 2s, 2p, 2s, 2p]
    exponents: np.ndarray (float)
        Gaussian function decay exponents for the primitives in the basis set.
    prim_per_shell: np.ndarray (integer)
        Array denoting the number of primitives per shell. If basis set is 6-31G this will be
        [6, 3, 3, 1, 1] if the atom is a heavy atom. This corresponds to
        [1s, 2s, 2p, 2s, 2p]. If additional atoms are present, the array is extended.
    coeffs: np.ndarray (float)
        Array of same length as `exponents` containing orbital expansion coefficients.
    """
    shells = []
    counter = 0
    # First loop over all shells
    for i, n in enumerate(prim_per_shell):
        shells.append(Shell(
            shell_map[i],
            [abs(shell_types[i])],
            ['p' if shell_types[i] < 0 else 'c'],
            exponents[counter:counter + n],
            coeffs[counter:counter + n][:, np.newaxis]
        ))
        counter += n
    del shell_map
    del shell_types
    del prim_per_shell
    del exponents
    del coeffs

    obasis = MolecularBasis(shells, CONVENTIONS, 'L2')
    return obasis


@document_load_one("MWFN", ['atcoords', 'atnums', 'atcorenums', 'energy',
                            'mo', 'obasis', 'extra', 'title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    inp = load_mwfn_low(lit)

    # MWFN contains more information than most formats, so the following dict stores some "extra" stuff.
    mwfn_dict = {'mo_sym': inp['mo_sym'], 'mo_type': inp['mo_type'], 'mo_numbers': inp['mo_numbers'],
                 'wfntype': inp['wfntype'], 'nelec_a': inp['nelec_a'], 'nelec_b': inp['nelec_b'],
                 'nbasis': inp['nbasis'], 'nindbasis': inp['nindbasis'], 'nprims': inp['nprims'],
                 'nshells': inp['nshells'], 'nprimshells': inp['nprimshells'],
                 'shell_types': inp['shell_types'], 'shell_centers': inp['shell_centers'],
                 'prim_per_shell': inp['prim_per_shell'], 'full_virial_ratio': inp['full_virial_ratio']}

    # Unlike WFN, MWFN does include orbital expansion coefficients.
    obasis = build_obasis(inp['shell_centers'],
                          inp['shell_types'],
                          inp['exponents'],
                          inp['prim_per_shell'],
                          inp['coeffs'],
                          lit)
    # wfntype(integer, scalar): Wavefunction type. Possible values:
    #     0: Restricted closed - shell single - determinant wavefunction(e.g.RHF, RKS)
    #     1: Unrestricted open - shell single - determinant wavefunction(e.g.UHF, UKS)
    #     2: Restricted open - shell single - determinant wavefunction(e.g.ROHF, ROKS)
    #     3: Restricted multiconfiguration wavefunction(e.g.RMP2, RCCSD)
    #     4: Unrestricted multiconfiguration wavefunction(e.g.UMP2, UCCSD)
    wfntype = inp['wfntype']
    if wfntype in [0, 2, 3]:
        restrictions = "restricted"
    elif wfntype in [1, 4]:
        restrictions = "unrestricted"
    else:
        raise IOError('No wfntype found, cannot determine if restricted or unrestricted wave function.')
    # MFWN provides number of alpha and beta electrons, this is a double check
    # mo_type (integer, scalar): Orbital type
    #     0: Alpha + Beta (i.e. spatial orbital)
    #     1: Alpha
    #     2: Beta
    # TODO calculate number of alpha and beta electrons manually.

    # Build the molecular orbitals
    mo = MolecularOrbitals(restrictions,
                           inp['nelec_a'],
                           inp['nelec_b'],
                           inp['mo_occs'],
                           inp['mo_coeffs'],
                           inp['mo_energies'],
                           None,
                           )

    return {
        'title': inp['title'],
        'atcoords': inp['atcoords'],
        'atnums': inp['atnums'],
        'atcorenums': inp['atcorenums'],
        'charge': inp['charge'],
        'obasis': obasis,
        'mo': mo,
        'nelec': inp['nelec_a'] + inp['nelec_b'],
        'energy': inp['energy'],
        'extra': mwfn_dict,
    }

def format_output(string, value, f, format="norm"):

    if np.dtype(value) == float:
        if format == "norm":
            print(str("{:10s}".format(string)) + str("{:15.6f}".format(value)), file=f)
        else:
            print(str("{:10s}".format(string)) + str("{:15.6e}".format(value)), file=f)
    elif np.dtype(value) == int:
        print("{:10s}".format(string) + "{:15d}".format(value), file=f)
    elif np.dtype(value) == str:
        print("{:10s}".format(string) + "{:15s}".format(value), file=f)
    elif value is None:
        IOError('Float, integer or string is required as formatted output.')
    #
    # print('{:10s} {}'.format('Charge=', data.charge or '?'), file=f)


def format_output_from_dict(word, value, obj, f):
    if obj.extra.get(value) is not None:
        print('{:12s} {:15d}'.format(word, obj.extra.get(value)), file=f)
    else:
        print('{:12s} {:15s}'.format(word, '?'), file=f)


def dump_mwfn_basis(basis, f):
    if basis.extra.get('nbasis') is not None:
        print('{:12s} {:15d}'.format('Nbasis=', basis.extra.get('nbasis')), file=f)
    elif basis.obasis.nbasis is not None:
        print('{:12s} {:15d}'.format('Nbasis=', basis.obasis.nbasis), file=f)
    else:
        print('{:12s} {:15s}'.format('Nbasis=', ' ? '), file=f)

    if basis.extra.get('nindbasis') is not None:
        print('{:12s} {:15d}'.format('Nindbasis=', basis.extra.get('nindbasis')), file=f)
    elif basis.obasis.nbasis is not None:
        print('{:12s} {:15d}'.format('Nindbasis=', basis.obasis.nbasis), file=f)
    else:
        print('{:12s} {:15s}'.format('Nindbasis=', ' ? '), file=f)

    if basis.extra.get('nprims') is not None:
        print('{:12s} {:15d}'.format('Nprims=', basis.extra.get('nprims')), file=f)
    else:
        print('{:12s} {:15s}'.format('Nprims=', ' ? '), file=f)

    print('{:12s} {:15d}'.format('Nshell=', len(basis.obasis.shells)), file=f)
    nprimshell = len([exps for shell in basis.obasis.shells for exps in shell])
    print('{:12s} {:15d}'.format('Nprimshell=', nprimshell), file=f)


@document_dump_one("MWFN", ['atcoords', 'atnums', 'atcorenums', 'mo', 'obasis', 'charge'],
                   ['title', 'energy', 'spinpol', 'lot', 'atgradient', 'extra'])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    line_length_cutoff = 78
    print(data.title + str(" (Created with IODATA)") or 'Created with IOData', file=f)
    print('{:10s} {}'.format('Charge=', data.charge), file=f)
    if data.extra.get('nelec_a') is not None:
        print('{:10s} {}'.format('Naelec=', data.extra['nelec_a']), file=f)
    else:
        print('{:10s} {}'.format('nelec_b', float(data.mo.norba)), file=f)
    if data.extra.get('Nbelec=') is not None:
        print('{:10s} {}'.format('Nbelec=', data.extra['nelec_b']), file=f)
    else:
        print('{:10s} {}'.format('Nbelec=', float(data.mo.norbb)), file=f)
    print('{:10s} {:15.8e}'.format('E_tot=', data.energy or '?'), file=f)
    if data.extra.get("full_virial_ratio") is not None:
        print('{:10s} {:15.8f}\n'.format('VT_ratio=', data.extra.get("full_virial_ratio")) or '?', file=f)
    else:
        print('{:10s} {:15s}\n'.format('VT_ratio=', '?'), file=f)

    print('# Atom information', file=f)
    print('{:10s} {:5d}'.format('Ncenter=', data.natom or '?'), file=f)
    print('# {}{}{}{:8}   {:8}   {:8}    {}'.format("|idx|", "sym|","num|", "nuc.chge|", "x coord    |", "y coord    |", "z coord    |"), file=f)
    for i in range(data.natom):
        num = data.atnums[i]
        sym = num2sym[num]
        nuc_chge = float(num)
        x, y, z = data.atcoords[i,0], data.atcoords[i,1], data.atcoords[i,2]
        print('{:>6d} {:2s} {:3} {:5} {:15.8f} {:15.8f} {:15.8f}'.format(i, sym, num, nuc_chge, x, y, z), file=f)

    if data.extra.get('nindbasis') is None:
        print('# Basis function information (IODATA has assumed Nbasis = Nindbasis)', file=f)
    else:
        print('# Basis function information', file=f)

    dump_mwfn_basis(data, f)

    print('$Shell types', file=f)
    try:
        string = '  '
        for shell_type in data.extra['shell_types']:
            string += str(shell_type) + str("  ")
            if len(string) > line_length_cutoff:
                print(string, file=f)
                string = '  '
        print(string, file=f)
        print('$Shell centers', file=f)
        string = '  '
        for center in data.extra['shell_centers']:
            string += str(center + 1) + str("  ")
            if len(string) > line_length_cutoff:
                print(string, file=f)
                string = '  '
        print(string, file=f)
        print('$Shell contraction degrees', file=f)
        string = '  '
        for prim in data.extra['prim_per_shell']:
            string += str(prim) + str("  ")
            if len(string) > line_length_cutoff:
                print(string, file=f)
                string = '  '
        print(string, file=f)
    except:
        print("Shell information not implemented currently", file=f)

    # Get exponents and contraction coefficients
    if data.obasis is None:
        raise IOError('A Gaussian orbital basis is required to write a MWFN file.')
    obasis = data.obasis
    exponent_list = []
    coeffs_list = []
    for shell in obasis.shells:
        for expi, coeff in zip(shell.exponents, shell.coeffs):
            exponent_list.append("{:e}".format(float(expi)))
            coeffs_list.append("{:e}".format(float(coeff)))
    if data.extra.get("nprimshells") is not None:
        number_primitive_shells = data.extra.get("nprimshells")
        assert len(exponent_list) == number_primitive_shells
        assert len(coeffs_list) == number_primitive_shells

    print('$Primitive exponents', file=f)
    # modify line length so limit is 5 items per row. Not necessary, but, mwfn does this.
    line_length_cutoff = 70
    string = '  '
    for expi in exponent_list:
        string += str(expi) + str("  ")
        if len(string) > line_length_cutoff:
            print(string, file=f)
            string = '  '
    print(string, file=f)
    print('$Contraction coefficients', file=f)
    string = '  '
    for coeff in coeffs_list:
        string += str(coeff) + str("  ")
        if len(string) > line_length_cutoff:
            print(string, file=f)
            string = '  '
    print(string, file=f)

    # Output molecular orbital information.
    print('\n# Orbital information (nindbasis orbitals)', file=f)
    if np.allclose(np.array(coeffs_list, dtype=float), np.ones(len(coeffs_list))):
        print('# Input file did not include contraction coefficients.', file=f)
        print('# IODATA has normalized the molecular orbitals to account for this.\n', file=f)
    else:
        # output blank line
        print('  ', file=f)
    default = [None for i in range(data.mo.norb)]
    if data.extra.get('mo_sym') is not None:
        syms = data.extra.get('mo_sym')
    else:
        syms = np.repeat('?', data.mo.norb)
    if data.extra.get('mo_type') is not None:
        mo_types = data.extra.get('mo_type')
    elif int(data.mo.norba + data.mo.norbb) == int(2 * data.mo.norb):
        mo_types = np.zeros((data.mo.norb), dtype=int) #[0 for i in range(data.mo.norb)]
    else:
        mo_types = default
    for i in range(data.mo.norb):
        print('{:10s} {:6d}'.format('Index=', i + 1), file=f)
        print('{:10s} {:6}'.format('Type=', mo_types[i]), file=f)
        # print('{:10s} {:6d}'.format('Type=', mo_types[i] or '?'), file=f)
        print('{:10s} {:8e}'.format('Energy=', data.mo.energies[i]), file=f)
        print('{:10s} {:15.6f}'.format('Occ=', data.mo.occs[i]), file=f)
        print('{:10s} {:}'.format('Sym=', syms[i]), file=f)
        string = '  '
        print(data.mo.coeffs.shape)
        for mo in data.mo.coeffs[:, i]: #data.mo.coeffs[i]:
            string += str("{:e}".format(mo)) + str("  ")
            if len(string) > line_length_cutoff:
                print(string, file=f)
                string = '  '
        print(string, file=f)
        print(' ', file=f)