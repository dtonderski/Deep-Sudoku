# DeepSudoku
A sudoku solving package using PUCT and AI. Inspired by the AlphaZero paper
"[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)".

## Algorithm
Sudokus are solved by using PUCT, which is a version of MCTS where rollouts are replaced with a 
neural network estimator. The key difference between DeepSudoku and AlphaZero is that each sudoku state
is either valid or invalid, whereas the lines are more blurred in chess.

For each move, a number of simulation iterations are run. Each iteration consists of exploring the move tree until a 
leaf or terminal node is hit. Then, the value (the probability of the node being valid) and the policy 
(the probability of each possible move being valid) is estimated by a neural network. The next move is based on
the result of those simulations.

## Training
At first, a Squeeze Excitation Resnet was trained using a loop with the following two stages:
1. Simulation

    A number of valid sudokus are initialized randomly. An attempt to solve them is then run using the most
    current network. As soon as an invalid move is played, the attempt is stopped, and all states encountered 
    during the search are saved. This is repeated until the number of encountered states reaches a predefined 
    number.
2. Training

    The encountered states of the last 10 simulation stages are used to train the neural network for a number
    of epochs. A key fact that makes the training process computationally feasible is that each sudoku can be
    augmented into ~10^12 other sudokus as described in the Sudoku section, and this process is computationally
    cheap. Each batch is augmented in this way - for extra efficiency, every sudoku in a specific batch is
    augmented identically.

This was not done using transformers as simulations using SeResNets have significantly more sudokus/second, so
data can be collected much quicker. After training has converged, the data from the previous 10 simulations is used
to pre-train a ViT-Ti based transformer network. This is done until convergence, and then the above training loop
is repeated for the transformer network.

Intuitively, sudokus with more blank cells should be more difficult, and so the training data should be biased toward
them for efficiency. To quantify this, a SeResNet was trained on sudokus where the number of empty cells was
sampled from a uniform distribution. The p loss was then calculated as a function of empty cells. This normalized 
quantity was used as the probability distribution of the number of empty cells for the training loop.

## Sudoku
It [has been shown](arxiv.org/abs/1201.0749) that a sudoku has to have at least 17 clues (initially filled cells) to have a valid and unique 
solution. The training data is based on the list of 49151 known 17-clue mathematically equivalent sudokus [published
by Gordon Royle](http://mapleta.maths.uwa.edu.au/~gordon/sudokumin.php). The website is currently down, but can be 
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

## Inputs, outputs, and loss function
The network takes as an input a batch of sudokus $x\in\mathbb{R}^{n_{batch}\times1\times9\times9}$, transforms it into a categorical tensor $x\prime\in\mathbb{R}^{n_{batch}\times9\times9\times9}$, and outputs a tuple $\hat{p},\hat{v}$, where $\hat{p}\in\mathbb{R}^{n_{batch}\times9\times9\times9}$, and $\hat{v}\in\mathbb{R}^{n_{batch}\times1}$. The target is a tuple $p,v$ of the same sizes as the output.

The loss function is computed as follows:

$L = l_p + l_v,$

$l_p = \sum_{i, v_i = 1}\sum_{rc, x_{irc} > 0}\mathrm{crossentropy}(p_{irc}, \hat{p}_{irc}),$

$l_v = \mathrm{binarycrossentropy}(v, \hat{v}).$

In words, the loss is the sum of:
1. $p$ loss, the crossentropy over the second dimension of $p$ and $\hat{p}$, only counting valid sudokus $(v_{i} = 1)$ and blank input cells $(x_{irc} > 0)$,
2. $v$ loss, the binary cross entropy of $v$ and $\hat{v}$.
