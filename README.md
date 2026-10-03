# Multi-Agent Package Delivery

A simulation where two robots compete on a 5x5 board to deliver packages and collect reward. 
Each robot runs one of three multi-agent algorithms, chosen
when the game starts. Robots lose battery with every move and die if it hits zero; they can recharge at charge stations using credit. 
Delivering a package to its destination earns credit, as a multiple of the delivery distance. 
At most two packages are on the board at once — a new one appears whenever one is delivered.

## The task
Implement the search algorithms and a heuristic function that maximizes packages delivered and reward collected, under battery and time constraints.

## Algorithms
- **Minimax**: assumes a fully adversarial opponent and picks the move that maximizes the worst-case outcome.
- **Alpha-beta**: minimax with pruning — skips branches that provably can't affect the final decision, without changing the result.
- **Expectimax**: assumes the opponent's moves are random rather than adversarial, weighted by how likely each move is
  (`EXPECTIMAX_ACTION_WEIGHTS`), and picks the move with the best expected
  outcome.

All three search to a fixed depth within a time limit, falling back to a default move if time runs out before a decision is reached.

## Notes
- Requires `func_timeout` (`pip install -r requirements.txt`).
- Needs the course's `WarehouseEnv` and `Agent`, which aren't included.

## Credits
built with Sarah Shaheen for intro to AI course, semester Spring 2025-2026.
