from tg2m.utils import data
from tg2m import utils
from tg2m.config.glob import *
import rdkit.Chem.rdchem.Mol

graphs = utils.smiles_to_molecular_graphs(f"{DATA_PATH}test.csv")
print(graphs[0])