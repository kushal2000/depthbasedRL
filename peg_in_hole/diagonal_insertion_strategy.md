# Diagonal-Approach Insertion: Prior Art & Relevance to Our Policy

Our latest peg-in-hole RL policy converges on a two-phase strategy: **approach the
hole diagonally (peg tilted), make edge contact with the hole rim, then
straighten the peg and push in vertically.** This is not a novel discovery of the
policy — it is, modulo phrasing, the classical "quasi-static insertion under
compliance" strategy first formalized by Whitney in the early 1980s and
observed in human manual-assembly studies since. This note collects citations
so we can frame the result honestly.

## 1. Why tilting first is mechanically optimal

The peg-in-hole task decomposes into two phases with qualitatively different
objectives ([ScienceDirect overview][overview]):

1. **Search** — find the hole despite positional uncertainty. A tilted peg
   reduces this phase to a one-dimensional problem: sweep the peg tip across
   the table until the tip edge catches the hole rim. A vertical peg instead
   has to align within the full clearance in both X and Y simultaneously.
2. **Insertion** — once the peg tip is seated on the rim, pivoting the peg
   upright naturally rolls the tip into the opening. The contact point at the
   rim acts as an instantaneous center of rotation.

The key property: a single edge-on-rim contact is a *one-point contact*
condition, which Whitney showed is stable under compliance. A vertical blind
descent that fails to hit the opening creates a *two-point contact* on the
outer surface, which is prone to **jamming** (peg stuck in a local friction
minimum) and **wedging** (peg deformed between two contact points, irrecoverable
without damage) ([Whitney 1982][whitney]; [Royal Society review of Whitney's
two-point / wedging analysis][royalsoc]).

Humans approximate this strategy naturally. Several robotic papers describe
their tilt-based controllers as "based roughly on the approach used by humans
in inserting tight-fitting parts" ([tactile feedback insertion strategy][tactile]).

## 2. Representative citations

### Foundational

- **Whitney, D.E. (1982).** *Quasi-Static Assembly of Compliantly Supported
  Rigid Parts.* ASME Journal of Dynamic Systems, Measurement, and Control,
  104(1): 65–77. Defined the mating events between peg and hole, derived the
  two-point-contact equilibrium, and introduced the Remote Center of
  Compliance (RCC). Established that the compliance center should sit at or
  near the peg tip so that a tilted peg pivots into the hole under applied
  axial force rather than jamming. [DOI 10.1115/1.3149634][whitney].
- **Simunovic, S. (1975/1979).** MIT thesis work directly cited as the basis
  for the RCC device; first formal analysis of tilt-based peg-hole alignment
  under uncertainty.

### Search + insertion decomposition

- **Chhatpar & Branicky (2001).** *Search Strategies for Peg-in-Hole Assemblies
  with Position Uncertainty.* IROS. Explicitly decomposes the task into a
  search phase (tilted-edge contact against the hole rim to localize) followed
  by an insertion phase (straightened descent). [ResearchGate][search].

### Explicit human-inspired tilt strategies

- **Tactile Feedback Insertion Strategy for Peg-in-Hole Tasks** — starts
  from a tilted peg and cites human tight-fit insertion as its basis; reports
  38/40 insertions successful on chamferless holes. [CORE PDF][tactile].
- **A Novel and Practical Strategy for the Precise Chamferless Robotic
  Peg-Hole Insertion** (Robotica, Cambridge). Uses a **6° tilt** for
  high-precision chamferless holes (0.1 mm clearance) with a wrist F/T sensor
  to detect edge contact. [Cambridge Core][chamferless].
- **Contact Pose Identification for Peg-in-Hole Assembly under Uncertainties**
  (arXiv 2101.12467). "First tilt, then rotation" CNN-based contact-pose
  classifier — explicitly a two-phase robotic implementation of the same
  human-observed pattern. [arXiv][contactpose].
- Other chamferless / inclined-hole assembly papers use tilt angles of
  **30°–45°** during the search phase to increase the probability of a peg-edge /
  hole-edge collision before insertion.

### Clearance and force ranges (for calibration)

- Manual assembly force studies put typical insertion pushes for small
  precision parts at **~15–30 N real**, rising to ~50 N for rubber-hose press
  fits. An acceptable ceiling for a 1 mm-clearance precision peg is roughly
  **30–50 N real** before damage concerns set in.

## 3. Relevance to our RL result

- The policy was trained end-to-end with goal-keypoint + retract rewards; no
  explicit "tilt" prior. That it converged on diagonal-approach-then-descent
  is a **convergence-to-known-optimum** result, not a novel strategy.
- The right framing for a paper / writeup: *"Our policy rediscovers the
  quasi-static insertion strategy formalized by Whitney (1982) and observed
  in human manual assembly — tilted rim-contact search followed by a
  compliance-assisted vertical descent — from scratch, without structural
  priors."* That's a defensible positioning.
- It also means any ablation showing the policy **fails to insert when forced
  to approach vertically** would be strong quantitative support for the
  mechanism, matching Whitney's jamming/wedging predictions.

[whitney]: https://doi.org/10.1115/1.3149634
[overview]: https://www.sciencedirect.com/topics/computer-science/peg-in-hole
[royalsoc]: https://royalsocietypublishing.org/doi/10.1098/rspa.2023.0364
[tactile]: https://files.core.ac.uk/download/pdf/595477908.pdf
[search]: https://www.researchgate.net/publication/3930400_Search_strategies_for_peg-in-hole_assemblies_with_position_uncertainty
[chamferless]: https://www.cambridge.org/core/journals/robotica/article/abs/novel-and-practical-strategy-for-the-precise-chamferless-robotic-peg-hole-insertion/DA07169AF5CB98EBE8158B980B74A284
[contactpose]: https://arxiv.org/pdf/2101.12467
