import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import collections
from numpy import vectorize as vec
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors, PandasTools
from rdkit.ML.Descriptors import MoleculeDescriptors

sdfFile =  'AID.sdf'
suppl = Chem.SDMolSupplier(sdfFile)

mol = suppl[0]

print mol.GetNumAtoms(), mol.GetNumBonds(), mol.GetNumHeavyAtoms()

print list(mol.GetPropNames())

print mol.GetProp('PUBCHEM_SUBSTANCE_SYNONYM')
