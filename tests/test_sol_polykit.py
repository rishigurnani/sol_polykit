import pytest
from rdkit import Chem

from sol_polykit import __version__
from sol_polykit import sol_polykit as spk
import warnings


def test_version():
    assert __version__ == "0.1.0"


@pytest.fixture
def example_polymer_data():
    return spk.LinearPol("[*]=CC=[*]")


def test_periodic_bond_type(example_polymer_data):
    assert example_polymer_data.periodic_bond_type == Chem.BondType.DOUBLE


def test_two_star_bonds():
    """
    This test checks that creating a LinearPol from the following SMILES
    string will fail. This should happen because one star
    contains two bonds.
    """
    sm = r"*CC/C=C\*(Cl)"
    # `match` takes a regular expression pattern, and some characters
    # like () are special. You need to escape them.
    sm_escaped = r"\*CC\/C=C\\\*(Cl)"
    with pytest.raises(
        ValueError,
        match=(
            f"Invalid repeat unit {sm_escaped}. "
            + "It is likely that at least one star contains more than one bond."
        ),
    ):
        spk.LinearPol(sm)


def test_mismatching_bonds():
    """
    This test checks that creating a LinearPol from the following SMILES
    string will fail. This should happen because one periodic bond is a
    single bond while the other periodic bond is a double bond.
    """
    sm = "*C1CCC(C=*)C1"
    # `match` takes a regular expression pattern, and some characters
    # like () are special. You need to escape them.
    sm_escaped = r"\*C1CCC\(C=\*\)C1"
    with pytest.raises(
        ValueError,
        match=(
            f"Invalid repeat unit {sm_escaped}. "
            + "Periodic bond types are mismatching."
        ),
    ):
        spk.LinearPol(sm)


@pytest.fixture
def example_ladder_data():
    sm = "C1C([g])C([e])OC([t])C1[d]"
    pol = spk.LadderPolymer(sm)
    return {
        "sm": sm,
        "pol": pol,
    }


def test_LadderPolymer(example_ladder_data):
    pol = example_ladder_data["pol"]
    # indices below are assigned as "A" since
    # 2 < 4
    assert pol.starA1_ind == 2
    assert pol.connectorA1_ind == 1
    assert pol.starA2_ind == 9
    assert pol.connectorA2_ind == 8
    # indices below are assigned as "B" since
    # 4 < 2
    assert pol.starB1_ind == 4
    assert pol.connectorB1_ind == 3
    assert pol.starB2_ind == 7
    assert pol.connectorB2_ind == 6


@pytest.mark.parametrize(
    "psmiles, mcsmiles",
    [
        # Linear polymer examples below.
        ("*CC*", "*CC*"),
        ("*C(C)C*", "*CC*"),
        ("*Cc1ccc(C*)c2cc(C)ccc12", "*Cc1ccc(C*)c2ccccc12"),
        # Ladder polymer examples below.
        # (
        #     "CCC1CC12C4C([e])C([d])CC3CC([g])C([t])C2C34",
        #     "",
        # ),
    ],
)
def test_MainChainMol(psmiles, mcsmiles):
    if "[g]" in psmiles:
        pol = spk.LadderPol(psmiles)
    else:
        pol = spk.LinearPol(psmiles)
    mcpol = pol.MainChainMol()
    result = Chem.MolToSmiles(mcpol.mol)
    assert result == mcsmiles, f"Resulting SMILES: {result}\nTrue SMILES: {mcsmiles}"


def test_ladder_smiles(example_ladder_data):
    alphasm = example_ladder_data["sm"]
    with pytest.deprecated_call():
        numericsm = spk.ladder_smiles(
            alphasm, e_repl="[200*]", t_repl="[201*]", g_repl="[301*]", d_repl="[300*]"
        )
    assert (
        spk.alpha2num_ladder_smiles(
            alphasm, e_repl="[200*]", t_repl="[201*]", g_repl="[301*]", d_repl="[300*]"
        )
        == numericsm
    )
    assert (
        spk.num2alpha_ladder_smiles(
            spk.alpha2num_ladder_smiles(
                alphasm,
                e_repl="[200*]",
                t_repl="[201*]",
                g_repl="[301*]",
                d_repl="[300*]",
            ),
            e_repl="[200*]",
            t_repl="[201*]",
            g_repl="[301*]",
            d_repl="[300*]",
        )
        == alphasm
    )
