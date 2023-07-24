t-SMILES: A Scalable Fragment-based Molecular Representation Algorithm for De Novo Molecule Generation
======================================================================================================

The directory contains source code of the article: Wu et al's Fragment-based
t-SMILES for de novo molecular generation

In this study, we propose a scalable, fragment-based, multiscale molecular
representation algorithm called t-SMILES (tree-based SMILES). The proposed
t-SMILES algorithm can build a multilingual system for molecular description, in
which each decomposition algorithm creates a kind of language, and all these
languages can complement each other and contribute to a whole mixed chemical
space.

![](media/4ce996a820a6223d29dac36b16f59343.jpg)

With t-SMILES, powerful and rapidly developing sequence-based solutions can be
applied to fragment-based molecular tasks in the same way as classical SMILES.

1.  Classical SMILES can be unified as a special case of t-SMILES to achieve
    better-balanced performance using hybrid decomposition algorithms;

2.  It significantly improves the generalization performance compared to
    classical SMILES, DeepSMILES, and SELFIES;

3.  It performs excellently on sparse datasets JNK3 and AID1706, regardless of
    whether it is the original model or based on data augmentation or pre-train
    fine-tuning;

4.  It outperforms previous fragment-based models being competitive with
    classical SMILES and graph-based approaches on Zinc, QM9 and ChEMBL.

Meanwhile, due to its unique structure, encoding and decoding algorithm,
t-SMILES possesses some distinguished properties, including:

1.  enabling a generative model without training and exploring a broader
    chemical space efficiently;

2.  being universally adaptable to any decomposition method such as BRICS,
    JTVAE, MMPA, or Scaffold;

3.  enabling the robust application of sequence-based generative models such as
    LSTM, Transformer, VAE, and AAE for molecular modeling.

![](media/8c47b4fad67f1ed36653f82b6c2447f5.jpg)

Here we provide the source code of our method.

Dependencies
============

We recommend Anaconda to manage the version of Python and installed packages.

Please make sure the following packages are installed:

1.  Python**(version \>= 3.7)**

2.  [PyTorch](https://pytorch.org/)** (version == 1.7)**

\$ conda install pytorch torchvision cudatoolkit=x.x -c pytorch

Note: it depends on the GPU device and CUDA tookit

(x.x is the version of CUDA)

1.  [RDKit](https://www.rdkit.org/)** (version \>= 2020.03)**

\$ conda install -c rdkit rdkit

1.  Networkx**(version \>= 2.4)**

\$ pip install networkx

1.  [Numpy](https://numpy.org/)** (version \>= 1.19)**

\$ conda install numpy

1.  [Pandas](https://pandas.pydata.org/)** (version \>= 1.2.2)**

\$ conda install pandas

1.  [Matplotlib](https://matplotlib.org/)** (version \>= 2.0)**

\$ conda install matplotlib

Usage
=====

For designing the novel drug molecules with t-SMILES representation, you should
do the following steps sequentially by running scripts:

1.  Applications/GPT2/GPT2App.py

>   train\_single\_voc\_file()

>   generate\_seq()

1.  DataSet/Graph/CNJMolAssembler.py

>   rebuild\_file()

In addition, this toolkit also provides some other scripts for data processing
and model architectures etc.

1.  DataSet/Graph/CNJTMol.py

>   It contained a preprocess function to generate t-SMILES from data set.

1.  DataSet/Tokenlizer.py

>   It defines a tokenizer tool which could be used to generate vocabulary of
>   t-SMILES and SMILES.

1.  Models/Parameters.py

>   It defines the parameter for training.

Acknowledgement
===============

We thank the following Git repositories that gave me a lot of inspirations:

1.  GPT2: <https://github.com/samwisegamjeee/pytorch-transformers>

2.  MolGPT: https://github.com/devalab/molgpt

3.  MGM：https://github.com/nyu-dl/dl4chem-mgm

4.  JTVAE：[https](https://github.com/wengong-jin/icml18-jtnn)://github.com/wengon-jin/icml18-jtnn

5.  hgraph2graph: https://github.com/wengong-jin/hgraph2graph

6.  FragDGM: https://github.com/marcopodda/fragment-based-dgm

7.  Guacamol：<https://github.com/BenevolentAI/guacamol_baselines>
