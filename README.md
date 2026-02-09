# NeuroGenesis

Neuromorphic brains grown from scratch with simulated evolution in Rust.

## Layout

- `sim-core`: deterministic simulation engine
- `sim-protocol`: shared protocol types
- `sim-server`: HTTP + WS simulation server
- `web-client`: HTML5 canvas renderer

## Quickstart

1. `cargo check --workspace`
2. `cargo test --workspace`
3. Start server: `cargo run -p sim-server`
4. In another shell: `cd web-client && npm install && npm run dev`
5. Open `http://127.0.0.1:5173`

## Config

A default config is available at `config/default.toml`.

## Concepts

### Neuron:

- List of neurotransmitter types released
- List of types of receptors at dendrites
- List of synapses
- Resting membrane potential
- Action potential threshold
- Membrane potential
- ID

### Receptor:

- List of matching neurotransmitters
- Behavior type (excitatory, inhibitory, modulatory)
- State (On/Off, time remaining)

Sensory Receptor:

- Behavior type
- Value

### Synapse:

- Synaptic strength
  - Probability of release
  - Number of neurotransmitters released

### Simulation Logic:

- Tick based
- Every tick (in order):
  - sensory receptors are reevaluated
  - all potential summations are recomputed
  - Action potentials are initiated
  - Axon neurotransmitters are released
  - Actions are taken if action neurons experience action potential
  - decay/propogation is incremented

Gene: - InterNeuron - Neurotransmitters - Receptors - MatchingNeurotransmitters

- potentialPolarization - Type - actionPotentialThreshold - restingPotential -
  PotentialDecayRate - Synapses - fireProbability - PreSynapticNeuron -
  PostSynapticNeuron

### Genome Encoding Structure:

- A genome is a sequence of Gene objects and a list of synaptic connections
  between all neurons, as well as parameters numNeurons and numSynapses
  - A gene corresponds to a Neuron or Synapse (is encoded as an array of floats)
- Creation Algorithm
  - Enumerate all InterNeurons, Sensory and Action neurons are enumerated
    according to the corresponding Enum
  - For each Neuron
    - Encode as a gene and append to Genome list
    - For each Synapse:
      - Encode as gene and append to Genome list Assumptions:
- Synapse transmission is instantaneous
- No local membrane potentiation in dendrites/neurons, membrane potential is
  global
- Action potential travel time is instantaneous
- Receptor only is activated for one tick by each neurotransmitter

## To Do:

- [ ] Decouple Neuronal/Brain Model from environment. Introduce interface for
      Brain
- [ ] When action potential fires, reduce potential harshly right afterwards
- [ ] Try negative neurons instead of synapses, less complexity
- [ ] Experiment with removing/reworking actionPotentialLength
- [ ] Come up with better solution for all the magic hardcoded constants (i.e.
      decay rate)
  - [ ] sensory normalization bound, max synaptic strength, survivor
        reproduction rate
- [ ] Implement culling useless neurons and synapses, neuronal pruning (also
      simulating exuberant synaptogenesis)
- [ ] Use Hebbian/SDTP to guide synaptogenesis (neurons that fire together wire
      together, neurons that fire out of sync lose their link). Synaptogenesis
      should not be random
- [ ] Implement mathematical models (differential equations, dynamic systems,
      analysis) for interactions like neural oscillations, electrotonic
      potential, cable theory, membrane potential decay, etc.
- [ ] Verify if Neurons is sorted normally by ID
- [ ] Reexamine actionPotentialTime tracking
- [ ] Reexamine encoding frequency based intensity of stimulus
- [ ] Reexamine order of computation for potentials in simulateStep
- [ ] Implement Fitness tracking
- [ ] Implement Neuron and Synapse monitoring per Organism
- [ ] Implement Neuron mutation
- [ ] Implement Adaptive Integrate-and-Fire model
      (https://en.wikipedia.org/wiki/Biological_neuron_model#Adaptive%20integrate-and-fire)
  - [ ] Read Neuronal Dynamics Ch 6.
        https://neuronaldynamics.epfl.ch/online/Ch6.S1.html
- [ ] Explore reinforcement learning instead of purely random search
- [ ] Add electrical gap-junctions
- [ ] Explore and implement oscillatory cyclic generation within neural networks
- [ ] Explore and implement neural plasticity
- [ ] Explore and implement information encoding schemes within neural signaling
      patterns
- [ ] Implement neural backpropogation of action potentials
- [ ] Explore NMDA(R)
- [ ] Why do axon terminals release nothing most of the time? Explore
      probability of release
- [ ] Implement modulatory synaptic behavior
- [ ] Implement keeping NeuronIDs consistent even with deleted neurons
