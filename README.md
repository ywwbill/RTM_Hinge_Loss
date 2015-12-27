# Relational Topic Model (RTM) with Hinge Loss
Code for model described in <http://cs.umd.edu/~wwyang/files/emnlp-2015-doclink.pdf> along with a sample dataset. 
Feel free to email me at wwyang@cs.umd.edu with any questions.

### Dependencies: 
- Java 1.8
- mallet.jar <http://mallet.cs.umass.edu/download.php>

### Commands to run:
- See main() in *RTM.java to run different RTM extensions
- See main() in SCC.java to identify strongly connected components in a given graph

### Input format (for your own data!):
- Vocabuary: Each line denotes a word.
- Corpus: Each line denotes a document in the following format
		len w_0:f_0 w_1:f_1 w_2:f_2 ... w_i:f_i ... w_n:f_n
where len denotes the length of this document; w_i is the index of the i-th word in vocabulary; f_i is the frequency of w_i in this document. Zero-frequency words can be omitted.
- Links: Each line denotes an edge and contains two integers separated by a tab (\t).

### Important hyperparameters:
- alphaSum (hyperparameter for document distribution over topics)
- regularization parameter c in *MedRTM.java (usually set to 1.0)
- negEdge and negEdgeRatio (option for sampling negative edges and the ratio of #neg-edges to #pos-edges, recommend to apply them)

### Other notes:
- The quality of clusters identified by strongly connected components may not be good when the graph is too dense/sparse. I'm working on a more robust model to solve this problem.
- Derivation of Gibbs sampling equation is available at <http://cs.umd.edu/~wwyang/files/emnlp-2015-doclink-supplementary-material.pdf>
- Seed words are not applied because they don't make much difference according to further experiments.
- If you use this code, please cite:

	@InProceedings{Yang:Boyd-Graber:Resnik-2015,
		Title = {Birds of a Feather Linked Together: A Discriminative Topic Model using Link-based Priors},
		Booktitle = {Empirical Methods in Natural Language Processing},
		Author = {Weiwei Yang and Jordan Boyd-Graber and Philip Resnik},
		Year = {2015},
		Location = {Lisbon, Portugal},
	}
