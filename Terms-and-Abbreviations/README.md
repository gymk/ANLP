# Terms and Abbreivations

## Similarity Measures

### Proximity Measure

A proximity[1] is a number which indicates how similar or how different two objects are, or are perceived to be, or any measure of this kind

### Tanimoto Coefficient

It is a similarity measure[2], can be applied to vectors having binary values as attributes.

Mostly Used in fingerprints[3] based similarity check. Especially in Cheminformatics.

Tanimoto Coefficient[4] value between 0 and 1, with 1 corresponding to identical fingerprints, i.e. protein–ligand interaction patterns

## LSI - Latent Semantic Indexing

LSI [5] compares how often words appear _together in the same document_ and compares how often those occurences happen in _ALL_ of the documents that Google has in its index
[6] Nice Explanataion

## Probability [7]

Probability is defined as the likelihood that an event will occur

Eg., Flipping a coin. There is a 50% chance or probability that heads will come up for any given toss of a fair coin.

Probability can be expressed as

- as a Percentage - Eg., 60%
- as a Decimal Form - Eg., 0.6
- as a Fraction - 6/10

### Probability in NLP

#### Why Probability used in NLP
- Probability will be used in estimating what could be the next word in the sentence
- Provides methods to predict or make decisions to pick the next word in the sequence based on sampled data
- Make the informed decision when there is a certain degree of uncertainty and some observed data
    - Example: How * you?
    - Finding all the possible words that might appeat in between How and you
    - To get an understanding, see Google NGram Viewer
- It provides a quantitative description of the chances or likelihood's associated with various outcomes

#### How Probability used in NLP

1. Probability of a Sentence
    - Probability of the next word in the sentence?
        - How likely to predict "you" as the next word after the query sentence "How are ____?"
            - Likelihood of the next word is formalized through an observation by conducting experiment - counting the words in a document

#### Discrete Sample Space
- Consider the following Bag of Words (_count = 52_)
    - Experiment
        - Extracting tokens from a document
    - Outcome
        - Every token/word in _x_ in the document
    - Sample Document
        - A weather balloon is floating at a constant height above Earth when it releases a pack of instruments. (Level 1) a. If the pack hits the ground with a downward velocity of −73.5 m/s, how far did the pack fall? b. Calculate the distance the ball has rolled at the end of 2.2 s 
- The outcome of the experiment - 52 sample (words).
    - They constitute the _sample space_, $\Omega$ or the set of all possible outcomes
        - $\Omega$ = 'a', 'weather', 'balloon', 'is', 'floating', 'at', 'a', 'constant', 'height', 'above', 'earth', 'when', 'it', 'releases', 'a', 'if', 'the', 'pack', 'hits', 'the', 'ground', 'with', 'a', 'downard', 'velocity', 'of', 'm', 's', 'how', 'far', 'did', 'the', 'pack', 'fall', 'b', 'calculate', 'the', 'distance', 'the', 'ball', 'has', 'rolled', 'at', 'the', 'end', 'of', 's'
- Each word in this sample belongs to $\Omega$, represented by $x \in \Omega$
- Eacm sample $x \in \Omega$ is assigned a probability score $[ 0, 1 ]$

#### Probability Mass Function
- Probability Function | Probability Distribution Function
    - A _probability function_ or _probability distribution function_ distributes the probability mass of $1$ to the all the samples in the sample space $\Omega$

#### Sample Space Constraints
- All the words in the $\Omega$, must satisfy the following constraints:
    1. $P(x) \in [0,1], for all x \in \Omega$
    2. $\sum_{x \in \Omega} P(x) = 1$

#### Events

- Events can be described as a variable taking a certain value
- An __*Event*__ is a collection of samples of the same type, $E \subseteq \Omega$
    - $P(E) = \sum_{x \in E} P(x)$
- Example
    - Consider above sample document
        - Total number of words = 52.
        - The number of _uniqye_ words = 37 or there are 37 __*types*__ of words in this BOW.
        - 15 words have frequencies $> 1$.

#### Random Variable

- A __random variable__[8], is a variable whose possible values are numerical outcomes of a random experiment
- Two types of random variable
    - Continuous
    - Discrete
- For NLP, it will be __*Discrete*__

- To capture Type-Token distinction, we use random variable $W$.
    - $W(x)$ maps to the sample $x \in \Omega$
- $V$ is the set of types and the value is represented by a variable $v$
- Given a random variable $V$ and a value $v$, $P(V = v)$ is the probability of the event that $V$ takes the value $v$, i.e: $P(V = v) = P(x \in \Omega: V(x) = v)$
    - Example: $P(V = 'the') = P('the') = 0.115$

----

## References

[1] <https://www.leydesdorff.net/aca/>

[2] <http://mines.humanoriented.com/classes/2010/fall/csci568/portfolio_exports/sphilip/tani.html>

[3] <https://www.surechembl.org/knowledgebase/84207-tanimoto-coefficient-and-fingerprint-generation>

[4] <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0302-y>

[5] <https://www.youtube.com/watch?v=LOPY1hPcZEM>

[6] <https://www.youtube.com/watch?v=OvzJiur55vo>

[7] <https://cs.brown.edu/courses/cs146/assets/files/langmod.pdf>

[8] http://www.stats.gla.ac.uk/steps/glossary/probability_distributions.html


- Markdown Symbols Help
    - <https://www.calvin.edu/~rpruim/courses/s341/S17/from-class/MathinRmd.html>
    - <https://csrgxtu.github.io/2015/03/20/Writing-Mathematic-Fomulars-in-Markdown/>
    - <https://github.com/Jam3/math-as-code/blob/master/README.md>

----
