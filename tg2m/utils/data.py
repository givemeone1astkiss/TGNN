import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

def standardize_smiles(smiles: str) -> None | str:
    """
    Standardizes a SMILES string by removing hydrogen atoms and sanitizing the molecule.

    Args:
        smiles (str): The SMILES string to standardize.

    Returns:
        str: The standardized SMILES string, or None if the standardization fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except TypeError:
        print(f"Failed to standardize SMILES: {smiles}")
        return None

def mol_to_nx(mol: Chem.Mol) -> nx.Graph:
    """
    Converts an RDKit molecule to a NetworkX graph.

    Args:
        mol (Chem.Mol): The RDKit molecule to convert.

    Returns:
        nx.Graph: The corresponding NetworkX graph.
    """
    adj_matrix = GetAdjacencyMatrix(mol)
    G = nx.Graph(adj_matrix)
    for i, atom in enumerate(mol.GetAtoms()):
        G.nodes[i]['atomic_num'] = atom.GetAtomicNum()
    return G

def smiles_to_molecular_graphs(csv_file_path: str) -> np.ndarray:
    """
    Reads a CSV file containing SMILES strings, standardizes them, and converts them to molecular graphs.

    Args:
        csv_file_path (str): The path to the CSV file containing SMILES strings.

    Returns:
        np.ndarray: An array of NetworkX graphs representing the molecular graphs.
    """
    df = pd.read_csv(csv_file_path)
    smiles_list = df['smiles'].tolist()

    molecular_graphs = []
    for smiles in smiles_list:
        standardized_smiles = standardize_smiles(smiles)
        if standardized_smiles:
            mol = Chem.MolFromSmiles(standardized_smiles)
            if mol:
                graph = mol_to_nx(mol)
                molecular_graphs.append(graph)

    return np.array(molecular_graphs)