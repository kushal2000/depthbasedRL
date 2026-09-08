# CAD → RL environment: what the pipeline automates

**Reviewer question (AC, R1, R3): how much of the CAD-design-to-RL-environment
pipeline is automated, and how much is common across problems?**

We assume the final CAD design in its **assembled pose**, as stated in the
paper. From that input the pipeline is fully automatic, with no human
intervention: the assembled pose is the goal pose; the removal direction and
thread axes come from the mating and threading geometry; the pre-contact pose is
the part retracted along that direction to the last contact-free configuration
(computed from its *swept* envelope, not a single cross-section); the receptacle
sits on the table by construction; the grasp box is the object's full extent;
collision geometry is SDF on contacting surfaces and convex decomposition
elsewhere; inertials come from mesh volume and density; and the goal tolerance
follows from the CAD's mating clearance (1 mm for the paper's tasks).

**Everything else is shared.** Adding a task touches no environment code: the
robot, observations, rewards, domain randomisation, termination, curriculum and
RL hyperparameters are identical across problems. Measured on the fork task
below, the per-task input is 73 lines of registration (object + problem entries)
and **7 changed lines** in a 231-line training config — the problem name, four
tolerances, and two logging tags. No changes to `isaacsimenvs/` at all.

**Evidence.** Two real-world-scale tasks were added this way with no pose
authored by hand: a YCB fork (197.6 × 27.1 × 15.9 mm) into a flatware holder at
1 mm clearance per side, and an Apple 20 W USB-C adapter (41.5 × 42.5 × 27 mm,
1.5 mm blades) into a NEMA 5-15R socket.

## Regenerating

```bash
python peg_in_hole_dynamic/plug_fork/create_assets.py --tolerances 0.5 1 2
python peg_in_hole_dynamic/plug_fork/viz_insertion_spec.py --port 8086   # inspect
```

`create_assets.py` writes the object and receptacle URDFs; `problems.py`
registers one `Problem` per clearance; `objects.py` registers the insertion
objects into `NAME_TO_OBJECT`.
