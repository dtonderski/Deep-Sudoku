# DeepSudoku
A sudoku solving package using Monte Carlo Graph Search and AI. Inspired by the AlphaZero paper
"[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)".

## Usage
To install the framework, simply run
```
pip install --upgrade --quiet git+https://github.com/dtonderski/DeepSudoku
```

For a demonstration on how to use the framework to solve a sudoku, see the Solving sudokus notebook.

## Novelties
1. Data - the goal of this project was to solve the most difficult sudokus - sudokus with 17 clues (i.e. filled cells). Most sudoku databases are based off of easier sudokus - for example, [this dataset](https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings) has at LEAST 19 clues.
2. Architecture - the best results were achieved using an domain-specific ViT-Ti based architecture, where we alternate running attention row-wise, column-wise, and block-wise. This results in drastically faster training compared to using cell-wise attention. Both attention-based method give drastic performance increase compared to a Squeeze Excitation ResNet.
3. Sampling - to generate training data, we run a tree search and save all the states encountered during this search. However, later on in the training, the vast majority of encountered states are trivial for the architecture, and do not "teach" it anything. To combat this, we only save states from games where the solver failed to solve the sudoku.
4. MCGS - I replace the Monte-Carlo Tree Search algorithm used by Alpha Zero with [Monte-Carlo Graph Search](https://arxiv.org/pdf/2012.11045.pdf). To explain
   the basic idea, consider moves $a_1$ and $a_2$. Starting from a sudoku state $s$, making moves $a_1, a_2$ results
   in the same position as making moves $a_2, a_1$. Utilizing this fact allows information exchange between subtrees,
   improving search performance. 

## Algorithm
Sudokus are solved by using Monte Carlo Graph Search with our predictor being a neural network estimator. The key difference allowing me to train this estimator in a reasonable time-frame is the following:

During training, we start from a sudoku position that has been solved deterministically, augment it, and try to solve
it using MCGS. Thus, we can easily tell if a state encountered during the search is valid by checking whether
all played moves are valid. Because of this fact, training the network is feasible on a consumer-grade GPU with a
network architecture identical to AlphaZero's, which used 5000 TPUs for training.

For each move, a number of simulation iterations are run. Each iteration consists of exploring the move tree until a 
leaf or terminal node is hit. Then, the value (the probability of the node being valid) and the policy 
(the probability of each possible move being valid) is estimated by a neural network. The next move is based on
the result of those simulations.

## Sudoku data
It [has been shown](arxiv.org/abs/1201.0749) that a sudoku has to have at least 17 clues (initially filled cells) to 
have a valid and unique solution. The training data is based on the list of 49151 known 17-clue mathematically 
non-equivalent sudokus [published by Gordon Royle](http://mapleta.maths.uwa.edu.au/~gordon/sudokumin.php). The website is 
currently down, but can be 
[accessed through Internet Archive's Wayback Machine](https://web.archive.org/web/20120722180233/http://mapleta.maths.uwa.edu.au/~gordon/sudokumin.php).
    
Mathematically equivalent sudokus mean sudokus that cannot be transformed into each other by any combination
of the following operations:
1. Permutations of the 9 symbols, 
2. Transposing the matrix (that is, exchanging rows and columns),
3. Permuting (ie. rearranging) rows within a single block, 
4. Permuting (ie. rearranging) columns within a single block, 
5. Permuting the blocks row-wise, 
6. Permuting the blocks column-wise. 

These operations allow a sudoku to be transformed in $9!\cdot6^8\cdot2\approx 10^{12}$ different ways.

Before the sudokus could be used for training neural networks, they were solved using the 
[py-sudoku package](https://pypi.org/project/py-sudoku/). Because of the slowness of this process (one sudoku could
take up to ~10 seconds), only the first 5800 sudokus were used. Because of the extremely powerful data augmentation
possibilities for sudokus, this did not cause any overfitting issues.

## Training
Intuitively, sudokus with more blank cells should be more difficult, and so the training data should be biased toward
them for efficiency. To quantify this, a SeResNet was trained on sudokus where the number of empty cells was
sampled from a uniform distribution. The average p loss was then calculated as a function of the number of empty cells. 
This normalized quantity was used as the probability distribution of the number of empty cells for the training loop.

Then, an initial Squeeze Excitation Resnet was trained using a loop with the following two stages:
1. Simulation

    A number of valid sudokus are initialized randomly from the above distribution. An attempt to solve them is then
    run using the most current network. As soon as an invalid move is played, the attempt is stopped, and all states
    encountered during the search are saved. This is repeated until the number of encountered states reaches a
    predefined number.
3. Training

    The encountered states of the last 10 simulation stages are used to train the neural network for a number
    of epochs. A key fact that makes the training process computationally feasible is that each sudoku can be
    augmented into ~10^12 other sudokus as described in the Sudoku section, and this process is computationally
    cheap. Each batch is augmented in this way - for extra efficiency, every sudoku in a specific batch is
    augmented identically.

Then, the data from the last 10 simulations is split into training, evaluation, and testing data,
which is used to evaluate different network architectures. The best network is then fine-tuned using 
the above training loop, but saving states from all runs, not just failed ones, as the network rarely fails
at this point.


The initial data collection and difficulty calculation was done using a SeResNet to make sure any bias 
tilts in favour of the SeResNet, not the transformer.





## Inputs, outputs, and loss function
The network takes as an input a batch of sudokus $x\in\mathbb{R}^{n_{batch}\times1\times9\times9}$, transforms it into a categorical tensor $x'\in\mathbb{R}^{n_{batch}\times9\times9\times9}$, and outputs a tuple $\hat{p},\hat{v}$, where $\hat{p}\in\mathbb{R}^{n_{batch}\times9\times9\times9}$, and $\hat{v}\in\mathbb{R}^{n_{batch}\times1}$. The target is a tuple $p,v$ of the same sizes as the output.

The loss function is computed as follows:

$L = l_p + l_v,$

$l_p = \sum_{i, v_i = 1}\sum_{rc, x_{irc} > 0}\mathrm{crossentropy}(p_{irc}, \hat{p}_{irc}),$

$l_v = \mathrm{binarycrossentropy}(v, \hat{v}).$

In words, the loss is the sum of:
1. $p$ loss, the crossentropy over the second dimension of $p$ and $\hat{p}$, only counting valid sudokus $(v_{i} = 1)$ and blank input cells $(x_{irc} > 0)$,
2. $v$ loss, the binary cross entropy of $v$ and $\hat{v}$.
