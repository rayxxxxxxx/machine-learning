# Road Search Algorithm (RSA) v0.0.1

This algorithm allows to find semi-optimal connections between graph vertices. Inspired by ACO (Ant Colony Optimization).

## Algorithm parameters

`Q` - constant, which divided by graph edges  
`alpha` - pheromone factor  
`beta` - edges value factor  
`gamma` - vertex priority factor  
`ph_r` (pheromone remain) - percentage of remain pheromone  
`pr_c` (priority coefficient) - vertex priority coefficient

You can modify parameters in `vars.py` file.  
Graph is `interactive`, you can move verticies.

## Actions

When pygame app is started:

- `[r]` button - reset pheromone
- `[p]` button - randomize vertices priority

## Requirements

- python3.10

Modules:

- pygame
- numpy
- pandas
- openpyxl

## Running:

> python3.10 main.py
