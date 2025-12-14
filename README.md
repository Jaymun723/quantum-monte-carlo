# Quantum Monte Carlo

## How to run the code

First make the production code available:
```sh
pip install -e .
```

The run a simulation located in the `design`folder:
```sh
python design/experiment_name.py
```

Data is stored in the `data` folder.

To visualise the results and generate figures, use notebooks the `analysis` folder.

## Project structure

```
quantum-monte-carlo
├── analysis
│   ├── energies.ipynb
│   ├── impact_of_n.ipynb
│   ├── J_x.ipynb
│   ├── loop_visualisation.ipynb
│   ├── temperature.ipynb
│   ├── thea.ipynb
│   └── trotter_error.ipynb
├── data
├── design  # experiment contains the name of the *fixed* parameters
│   ├── local_n_J_x_J_z_T.py
│   ├── loop-J-x-J-z-T-m.py
│   ├── loop_J_x_J_z_T_m.py
│   ├── loop-n-J-x-J-z-m.py
│   ├── loop_n_J_x_J_z_m.py
│   ├── loop_n_J_x_J_z_T.py
│   ├── loop_n_J_z_T_m.py
│   └── vertex_n_J_x_J_z_T.py
├── figures
├── production
│   ├── exact.py  # Code for exact computations
│   ├── exhaustive_worldline.py  # Generates all worldlines
│   ├── __init__.py
│   ├── local_updates.py  # local shifts
│   ├── loop_updates.py
│   ├── monte_carlo.py  # performs the whole simulation
│   ├── problem.py  # Class to holds parameters
│   ├── utils.py
│   ├── vertex_updates.py
│   └── worldline.py
├── README.md  #this !
├── report  # source code of the repport
└── setup.py  # to make the folder `production` available
```