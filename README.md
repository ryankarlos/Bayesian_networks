# Probabilistic Graphical Models

Bayesian networks are graphical models where nodes represent random variables and arrows represent probabilistic
dependencies between them. The graphical structure of a Bayesian network is a directed acyclic graph (DAG) which
define the joint probability distribution of V = (X1, X2,..., XX). This can be factorized into a set of local probability
distributions, one for each variable.

Bayesian networks are exceptionally flexible when doing inference, as any subset of variables can be observed, and
inference done over all other variables, without needing to define these groups in advance. So a set of observed variables
can change from one sample to the next without needing to modify the underlying algorithm at all.

The learning task of a Bayesian network consists of three subtasks:

* Structural learning, i.e. learn the graphical structure of the Bayesian network
* Parametric learning, i.e. estimate the parameters of the local distribution functions conditional on the
learned structure
* Running Inference queries Le evaluate the distributions of other variables of interest given a subset of observed variables

* Python provides a number of well known libraries for creating Bayesian Networks e.g.pomegranate, pgmpy. These provide
options for initialising our own network using domain knowledge or learning the structure and conditional probability
tables (CPD) from the data. Note that there are some limitations with these libraries which may only implement
discrete Bayesian networks (with the exception of pampy, although slightly more complex to implement, which require
any continuous variables to be discretized. Additionally, initialisation and parameter fitting can be computationally
challenging as number of nodes (dimensions) in the dataset increases including number of observations.

We can estimate a DAG that captures the dependencies between the variables. To learn model structure, there are two
broad techniques: score-based structure learning  and constraint-based structure learning. The combination of both techniques
allows further improvement i.e. hybrid structure learning.


## Structure Learning

### Score-based Structure Learning

This approach construes model selection as an optimization task. It has two building blocks:

* A scoring function that maps models to a numerical score, based on how well they fit to a given data set.

Commonly used scores to measure the fit between model and data are:
1. Bayesian Dirichlet scores such as BDey or K2
2. Bayesian Information Criterion (BIC, also called MDL).

* A search strategy to traverse the search space of possible models and select a model with optimal score.  The search
space of DAGS is super-exponential in the number of variables and the above scoring functions allow for local maxima.
The first property makes exhaustive search intractable for all but very small networks, the second prohibits efficient
local optimization algorithms to always find the optimal structure. Thus, identifiving the ideal structure is often
not tractable. Despite this, heuristic search strategies often yields good results.

*Algorithms*

1. Chow-Liu tree-building algorithm: This algorithm first calculates the mutual information between all pairs of
 variables and then determines the maximum weight spanning tree through it. The pseudocode can be roughly
broken down into the following steps:

- Compute all pairwise mutual information
- Find a maximum spanning tree of the undirected, fully connected graph on V with edge weight IQ(XX; Xu) between node v and u.
Repeatedly select an edge with maximum weight that does not create a cycle.
- Make any node of the spanning tree as the root and direct edges away from it.  The result is a rooted tree `G` that
maximizes `p(G)`.

2. The Tree Augmented Naive Bayes Algorithm (TAN) build a Bayesian network that is focused on a particular target
T (eg for classification). The algorithm creates a Chow-Liu tree, but using tests that are conditional on T e.g.
conditional mutual information. Then additional links are added from T to each node in the Chow-Liu tree

3. HillclimbSearch implements a greedy local search that starts from the DAG start (default: disconnected DAG) and
proceeds by iteratively performing single-edge manipulations that maximally increase the score. The search
terminates once a local maximum is found.

### Constraint-based Structure Learning

A different, but quite straightforward approach to build a DAG from data is this:
1. Identify independencies in the data set using hypothesis tests
2. Construct DAG (pattern) according to identified independencies

*Conditional Independence Tests*

Independencies in the data can be identified using chi2 conditional independence tests. To this end, constraint-based
estimators in pgmpy, have a test conditional independence (X, Y, Z)-method, that performs a hypothesis test on the
data sample. It allows to check if x is independent from y given a set of variables pgmpy package provides the following
conditional independence test options:

1. Chi-Square test (https://en.wikipedia.org/wiki/Chi-squared test)
2. Pearson (https://en.wikipedia.org/wiki/Partial correlation#Using linear regression)
3. G-squared (https://en.wikipedia.org/wiki/G-test)
4. Log-likelihood (https://en.wikipedia.org/wiki/G-test)

For example, using chi squared test the following example returns a boolean whether the variables are independent or not
(given the evidence) - based on the computed chi2 test statistic, the value of the test (and significance level set).
The value is the probability of observing the computed chi2 statistic (or an even higher chi2 value), given the null
hypothesis that X and Y are independent given `Zs`.

```
from pgmpy estimators import ConstraintBasedEstimator

data = pd.DataFrame(np.random.randint(0, 3, size=(2500, 8)), columns=list('ABCDEFGH'))
data['A'] += data['B'] + data['C']
data['H'] = data['G'] - data['A']
data['E'] *= data['F']
est = ConstraintBasedEstimator(data)

def is_independent(X, Y, Zs=[], significance_level=0.05):
    return est.test_conditional_independence(X, Y, Zs)

print(is_independent('B', 'H')) #False
print(is_independent('B', 'E')) #True
print(is_independent('B', 'H', ['A'])) #True
print(is_independent('A', 'G')) #True
print(is_independent('A', 'G', ['H'])) #False
```

*DAG (pattern) construction*

With a method for independence testing at hand, we can construct a DAG from the data set in three steps:

1. Construct an undirected skeleton-satinate skeleton()
2. Orient compelled edges to obtain partially directed acyclic graph (PDAG; I-equivalence class of DAGs) - skeleton__ta_edag!)
3. Extend DAG pattern to a DAG by conservatively orienting the remaining edges in some way - pag_dag()

Step 1.&2. form the so-called PC algorithm, see [2], page 550. PDAGs are DirectedGraphs, that may contain both-way edges,
to indicate that the orientation for the edge is not determined.

### Hybrid Structure Learning

The MMHC algorithm (Tsamardings et al., The max-min hill-climbing BN structure learning algorithm, 2005) combines
the constraint-based and score-based method. It has two parts:

1. Learn undirected graph skeleton using the constraint-based construction procedure MMPC
2. Orient edges using score-based optimization (BDeu score + modified hill-climbing)

#### Implementing Domain Knowledge

If you readily know (or you have domain knowledge) of the relationships between variables, we can setup the (causal)
relationships between the variables with a directed graph (DAG). Each node corresponds to a variable and each edge
represents conditional dependencies between pairs of variables., where the second value in each tuple is the variable
which is dependent on the first.

In the example below, for a health diagnosis - we can define define dependencies of lung cancer/bronchitis on whether
someone smoking or not. Similarly, the probability of x ray result being positive or negative is dependant on
whether someone has lung cancer/bronchitis

```
edges = [('smoke', 'lung'), ('smoke', 'bronc'), ('lung', 'xcax'), ('bronc', 'xrax')]
```

Building the DAG from first principles:

1. Create an undirected weigted network x graph based on weight metric computed (adjusted mutual info).
Use pgmpy static method of Treesearch Class to return only weights based on adjusted mutual.info or Normalized mutual info.

```
from pgmpy estimators import TreeSearch
import networkx as ox
import pandas as pd

def get_mutual_info_weights(df, edge_weights_fn= "mutual_info"):
    return TreeSearch._get_weights(df, edge weights.fn. show progress=False)

def create nxundirected weighted(weights.dt):
    net =
    Returns undirected weigted network x graph based on weight metric computed
    cols=df.columns
    return nx.from_pandas_adjacency(pd.DataFrame(weights,index=cols, columns=cols), create_using=nx.Graph)

weights = get_mutual_info_weights(df)
net = create_nx_undirected_weighted(weights, df)
```

2. Implemented must algorithm for pruning an undirected weighted network. Starts by sorting edge weights in descending
order and adding to nodes in unconnected network. If cycle formed - then edge is discarded.

```
def make_acyclic_pruned_network(net):
    return nx.maximum_spanning_tree(net)

pruned.net = make_acyclic_pruned_network(net)
plt.figure(figsize=(15,15))
nx.dcaw.networkx(pruned.net, pos=nx.spring_layout(dag))
```

3. Use *Breadth First Search (BFS)* for traversing the unweighted graph. It uses the following steps:

- BFS starts with the root node and explores each adjacent node before exploring node(s) at the next level.
- BFS makes use of *Queue* for storing the visited nodes of the graph / tree.

```
from networkx.algorithms.traversal.breadth_first_search import bfs_tree

dag = bfs_tree(pruned net source='COL_NAME')
plt.figure(figsize=(15,15))
nx.draw_networkx(dag, pos=nx.spring_layout(dag))
```

## Parameter Estimation

Fitting a Bayesian network to data is a fairly simple proces using pomegranate or balear, Essentially, for each
variable, you need consider only that column of data and the columns corresponding to that variables parents. If it
is a univariate distribution, then the maximum likelihood estimate is just the count of each symbol divided by the
number of samples in the data. If it is a multivariate distribution, it ends up being the probability of each
symbol in the variable of interest given the combination of symbols in the parents.

For example, consider a binary dataset with two variables, X and Y, where X is a parent of Y. First, we would go through
the dataset and calculate `P(X=0) and P(X=1)`. Then, we would calculate `P(Y=0|X=0)`, `P(Y=1|X=0)`, `P(Y=0|X=1)`,
and `P(Y=1|X=1)`. Those values encode all the parameters of the Bayesian network.

## Inference

We can then use or model of network to carry out inference which is same as asking conditional probability questions to
the models. e.g. What is the probability of the alarm being on given one is burgled. So we can compute the probability
of an event given the evidence observed. Currently, pampy support two algorithms for inference, both of these are
exact inference algorithms.

1. *Variable Elimination*: save computing time and avoid computing the full Joint Distribution by doing marginalization
over much smaller factors. So basically if we want to eliminate X from our distribution, then we compute the product
of all the factors involving X and marginalize over them, thus allowing us to work on much smaller factors.

2. *Belief Propagation*.

In the case of large models, or models in which variables have a lot of states, inference can be quite slow. Some
ways to deal with it are:

1. Reduce the number of states of variables by combining states together.
2. Try a different elimination order or custom elimination order.
3. Approximate inference using sampling algorithms.


## Dynamic Bayesian Networks

Dynamic Bayesian networks (DBNs) are an extension of Bayesian networks to model dynamic processes. A DBN consists of a
series of time slices that represent the state of all the variables at a certain time, t. We can define an initial
distribution over states BO (base network structure) and copy this structure over successive time steps T. Based on
the stationarity property, a 2-TBN defines the probability distribution P(X(t+1) | X(t)) for any t.

Given a distribution over the initial states, we can unroll the network over sequences of any length, to define a
Bayesian network that induces a distribution over trajectories of that length. where:

- for any random variable `Xi(1:n)`, all the copies of the variable `X(t)i for t> 0` have the same dependency structure
and the same CPD.

Thus, we can view a DBN as a compact representation from which we can generate an infinite set of Bayesian networks
(one for every T > 0). Pgmpy implementation of DBN used this approach Dynamic Bayesian Network (DBN), although the
documentation seems slightly unclear with regards to estimation of cod from data.
For learning the base structure we can use all the available data for each variable, ignoring the temporal information.
This is equivalent to learning a BN. We can then consider temporal information for each state by adding n copies
of the structure learnt above for n timesteps (in this case we only do it for 2 time steps Xt and Xt+1) and
adding edges between nodes in previous time slice and next.
