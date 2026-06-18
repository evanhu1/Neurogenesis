---
type: Mechanism
title: plant_consumption_rate ≈ metabolism / food_energy (and food_energy conflicts with learning)
description: Foraging rate tracks how often organisms must eat; lowering food_energy raises it but starves them.
confidence: high
supported_by:
  - experiments/0001-food-lower-food-energy
tags: [food-ecology, foraging, food-energy]
timestamp: 2026-06-16T00:00:00Z
---

# Mechanism

Carried prior, re-confirmed this iteration. `plant_consumption_rate` tracks how
often an organism must eat ≈ `metabolism / food_energy`. **Lowering food_energy is
the dominant foraging lever** — [[experiments/0001-food-lower-food-energy]] raised
plant to 0.087–0.094 — **but it conflicts with learning/intelligence**: per-bite
energy cuts starve organisms, cratering action_effectiveness (seed 7 AE
0.443 vs 0.566 baseline).

Corollary: foraging toward the 0.10 target should be pursued via **findability**
(plant abundance / regrowth speed) layered on a **soft survival economy**
(homeostatic metabolism), NOT via food_energy cuts. Findability alone gave only
modest gains (plant 0.066–0.072) at some action_effectiveness cost
([[directions/food-findability-with-hold-pillar-guard]]), so 0.10 likely needs a
*combination* (findability + the homeostatic champion), tested explicitly.
