# Stage F spec-review findings (2026-10-07, NOT yet dispositioned)

Two adversarial critics (Opus 5.5 xhigh) reviewed the draft in this folder.
The reviser was stopped mid-way when the work was parked, so the draft
files may carry SOME of these fixes; re-check each against the draft.

## critic A

Adversarial review of the Stage F draft (method and numerics lens). I read it against the code at f297867e and the working tree, the frozen D21.5 and D21.9.5 contracts, the prototype scripts and logs, and Ernould 2020 and 2021. I edited no repo file. I ran two small CPU-only scripts on existing data: stageF_critic/subbatch_amp.py and stageF_critic/dilution.py in the session scratchpad.

**Verified correct:**
- **Window stencil.** The periodic-Hann frequency stencil `X/2 - (X[k-1] + X[k+1])/4` is exact.
- **Angle profile and search window.** The magnitude of a real image is point symmetric, so the polar profile repeats every pi and a length-360 circular correlation is exact. The +-30 deg window removes the 180 deg ambiguity. It also excludes the 60 and 90 deg pseudo-symmetry aliases of 3-, 4- and 6-fold zone axes.
- **Parabolic peak.** The parabolic sub-bin formula is correct and continuous across a bin switch.
- **Angle sign.** The rule h21 = +sin(theta) is consistent with D1.5 and with the spectrum sampling `(fx sc, fy sr)`.
- **Starting row.** The partial row `W0 = R(theta) T(t)` is exact for any rigid motion, including a projection-centre shift: `t = R^-1 gamma` gives `W0 xi = R xi + gamma`. It matches Ernould 2021 eq. E.1.
- **Gate twist formula.** `atan2(R10 - R01, R00 + R11) = 2 atan2(q_z, q_w)` is the swing-twist twist, and it equals the homography in-plane read-out for a lattice rotation. The conjugation `R_det = M R_s M^T` is correct.
- **FM17 mutant.** It is exactly equivalent on an untilted detector (`M M = I`), so the tilted G7 detector the draft specifies really is needed.
- **D21.5 seam.** The NaN-row rule is respected.

**Main problems, in order:**
1. The real-data evidence for acceptance and gain was measured with the complete row, which D22 rejects. The frozen partial row was never fit on real data with the frozen recipe, nor beyond 5 deg.
2. The headline capture numbers ("113 cases to 30 deg", "-6 to 10 deg real", "captures at least 30 deg") come from rejected recipes. The +-30 deg window also makes "at least 30" impossible.
3. The required real-data frame check (F13), which can stop the stage, has an ill-posed decision rule. On the same 32 points the slope is 0.45 one way and 1.55 the other (regression dilution).
4. Several test oracles are likely wrong or too weak:
   - V10(f) expects the 8 deg ramp point to keep its `"off"` outcome, but the draft's own data suggest it will converge from an FM seed.
   - The pin that should kill the bin-unit look-up-table mutant (FM3) conflicts with the 2x-margin rule, so FM3 likely survives.
   - The G5 and G8 premises are incomplete.
5. Writing output keys into `SeedBatch.extras` breaks the wording of D21.5 clause (iv), and a purely additive alternative exists.
6. The GPU cost is stated per routed pattern, but the masked full-P branch costs per slot of every sub-batch that contains a routed point. On the Si map that is a 5.2x amplification and about 4 s, roughly 2.5 per cent of the map run. Acceptable, but it should be stated per slot.

The minor items are spec gaps (tie order, comparison with a non-finite criterion, the GPU forced-slot refit, the D21.12 message wording), cost and memory accounting, the angle with a dead-band cross, and a citation off by two lines.

### F01 (major) -- requirements_D22.md D22.5 'Measured (ledger 214)' paragraph and D22.4.3; validation_V10.md ledger 214 (iii)-(iv); proto/e2e.py lines 151-158 and 191-195

Issue: The frozen configuration is the partial row with the V8 recipe and the acceptance check, and it has no real-data or large-angle end-to-end measurement. Every real-rim fit and acceptance count quoted in D22.5 compares T against the COMPLETE row C: 13 of 16 accepted, 1616 against 2002 iterations, and 5 of 8 converted under (0.05, None). D22.4 rejects that row. e2e.py also fits only from C in the projection-centre-shift arm B (`fit(stateB, target, rows['C'], 50)`), so partial-row fits beyond 5 deg were never run with the D22 recipe. The partial-row rim data (ledger 206) come from a different recipe: periodic amplitude with rho <= 0.20.

Fix: Before freezing D22.4 and D22.5, re-run e2e.py's real-rim arms at both filter settings and its arm B with the partial row P and the acceptance check. This is about 2 CPU minutes, single process. Restate the D22.5 'Measured' paragraph and ledger 214 (iii)-(iv) with P's numbers. If that cannot happen at this gate, label the current numbers 'complete form only' and list the partial-row real-data record as a required implementation-gate measurement in plan 12 item 3.

### F02 (major) -- requirements_D22.md D22 lead paragraph ('113-case synthetic sweep up to 30 deg', '-6 to 10 deg'), D22.16 docstring Notes ('captures at least 30 deg'); validation_V10.md ledger 202(ii), 203(ii), 215, V10(e) weekly 30 deg arm; roadmap_stageF.md

Issue: The capture numbers come from recipes D22 rejects. sweep.py and realrot.py use `fm.cfg_of()`: whitened per-radius phase correlation, log with median scaling, 64 radii and the two-candidate 'both' arbitration that D22.3.7 refuses. sweep2.py uses fm.REC, which is variant V1 with the nearest rule and no +-30 window. The frozen V8 recipe was fit end to end only up to +-5 deg (e2e A) and seed-level to 15 deg (e2e B, fitted from C). With a +-30 deg search window a capture beyond about 30 deg is impossible by construction, so 'at least 30 deg' cannot be true. A 30 deg case sits exactly on the window edge, where the argmax is clamped.

Fix: Tag every capture number in the lead, ledger 202/203/215 and the roadmap with the recipe that produced it. Change the docstring claim to 'up to the 30 deg search window; with this recipe measured to N deg', where N comes from V10(e). At the failing-tests gate, run the border-0.15 arms (8, 10, 15, 20 deg) on the CPU with V8, the partial row and the acceptance check before quoting them. Use 25 deg rather than 30 deg for the weekly arm, or add an explicit window-edge arm.

### F03 (major) -- requirements_D22.md D22.7 'ON REAL DATA THE FRAME IS NOT YET PINNED'; validation_V10.md V10(m)(1), ledger 212(v); plan_section12.md F13 ('a slope far from +1 stops the stage')

Issue: The frame check that can stop the stage has an ill-posed decision rule. The regressor is the noisy Hough twist, so a least-squares slope of fitted-on-xmap is attenuated even when the frame is correct. Selecting points with |xmap twist| >= 1 deg on that noisy value adds regression to the mean. Measured on the same 32 rim points (realdata2.json): fitted-on-xmap slope 0.452, xmap-on-fitted slope 1.545, geometric-mean slope 0.54, median ratio 1.19. The standard deviation of the xmap twist is 0.221 deg against 0.120 deg for the fitted rotation. The data bracket 1 and are consistent with a correct frame plus Hough noise. The spec also does not say which mode produces the converged homographies. Under 'auto', the frame being tested decides which large-twist points get FM and converge, which biases the sample.

Fix: Take the converged homographies from an `"always"` run, which is independent of the xmap. Select on the FITTED in-plane rotation (|y| >= 1 deg), not on the xmap twist. Regress xmap twist on fitted rotation, or use Deming regression with the Hough variance measured on the far field. Base the frame decision on comparing correlations across candidate frames: as-read, +-90 deg and 180 deg about sample z, and M without the y flip. The as-read frame must give the highest correlation and a slope within a stated band such as [0.8, 1.25]. Record the reverse slope (1.545) in ledger 212(v) now.

### F04 (major) -- validation_V10.md V10(f) ('the 8 deg out-of-frame point and the constant failed pattern keep their "off" outcomes')

Issue: The ramp map's unfittable point (RAMP_UNFITTABLE_INDEX = 15, an 8.0 deg in-plane rotation) was measured to fail only from the phase-XC seed. Under `"always"`, and under `"auto"` with a matching xmap (twist 8 deg, so routed), it gets a near-exact FM row. Ledgers 202(i) and 215 show that 8 deg converges from an FM or exact seed with an error at the mirror-boundary scale (0.16 px at PC_480, border 0.05). The V8 PC is in frame to 6.62 deg, so the point most likely converges, and the 'keep off outcome' assertion will fail or push a wrong pin. The 9 deg out-of-plane rescue point (index 17) also gets an FM row and the acceptance check, but its expected outcome is not specified.

Fix: For index 15, record the outcome rather than asserting it equals `"off"`: expected converged with an error at the ledger 215 mirror-boundary scale, failing the D2.6 contract only if measured so. Specify index 17 as well: the accepted row and its convergence recorded, or asserted converged. Keep the constant-pattern point's D2.6 assertion.

### F05 (major) -- validation_V10.md V10(b) G3 arm ('pinned BELOW the 2.3 per cent ... 0.18 deg at 8 deg'); plan_section12.md FM3 ('the ONLY killer')

Issue: The pin rule for FM3's sole killer contradicts itself. The bin-unit look-up-table mutant is off by 0.185 deg at 8 deg, yet every band is 'pinned at about 2x margin' of the recipe's own measured error. V8's noise-free error on G1 is up to 0.123 deg (h2h.log, SYN clean), so a 2x pin of about 0.25 deg lets FM3 survive. The 0.002-0.012 deg physical-frequency errors of ledger 208 came from a different recipe (amplitude, rho <= 0.20, real pattern), not V8 on G3.

Fix: Run the G3 arm at seed level only (an angle-only test needs no in-frame margin) at 15 and 20 deg, where the mutant's bias is 0.35-0.46 deg. Measure V8's G3 error at the failing-tests gate and state the kill separation beside the pin. Optionally use a more elongated crop to widen the gap.

### F06 (major) -- requirements_D22.md D22.6 ('SeedBatch.extras OUTPUT keys'), BLOCK 6 (D21.5 amendment), BLOCK 7

Issue: D21.5 clause (iv) allows new fields and 'new extras keys that the runner fills from host data keyed by pattern_index'; existing fields never change. D22 instead has the seed stage WRITE `fourier_mellin_angle` and `fourier_mellin_applied` into `batch.extras`, as ctx.xp (device) arrays beside the host int8 input key, and amends D21.5 to allow it. That changes the contract of an existing frozen field, which Johan's decision 1 exists to avoid. It also turns a shared input dict into a side channel: any spy that keeps a reference to `extras` sees it change after the call.

Fix: Add a new SeedBatch field, for example `outputs` (a dict, default empty, written only by `seed_homographies` and holding ctx.xp arrays), which clause (iv) permits as a 'new field'. Keep `extras` input-only and on the host. BLOCK 6 then becomes purely additive (the new field plus the new input key). The V9 `extras == {}` assertions stay true for every mode.

### F07 (minor) -- requirements_D22.md D22.15 ('about 2.2 to 2.5 ms per routed pattern'); plan_section12.md 12.1 risk 5, F9

Issue: Because the FM branch runs masked at the full sub-batch size P (D22.6), the device cost is per SLOT of every sub-batch that holds at least one routed point, not per routed pattern. Measured on the gate twist map, in map order with the CCC < 0.35 crater masked and P = 32: at 1.5 deg, 336 routed points touch 55 of 1806 sub-batches, 1760 slots (5.2x); at 2.0 deg, 73 routed give 544 slots (7.5x); at 1.0 deg, 1012 routed give 3552 slots (3.5x). At 2.2-2.5 ms per slot that is about 4 s, roughly 2.5 per cent of the 166.5 s map. Under `"always"` it is about +130-145 s, roughly +80 per cent, which matches the seed's 62 per cent share.

Fix: Restate D22.15, risk 5 and F9 per slot of a routed sub-batch, quoting these counts (script stageF_critic/subbatch_amp.py). Add the `"always"` whole-map expectation.

### F08 (minor) -- requirements_D22.md D22.15 ('the whole CPU FM seed about 66 to 101 ms ... together about five IC-GN iterations'), D22.10

Issue: The CPU route cost leaves out work D22.10 itself requires. A routed point is preprocessed and splined once for the P = 1 seam and again inside `fit_pattern` (band-pass 35.5 ms plus spline 9.9 ms at 512x622, ledger 90). It computes h_T through the numpy seam (the `initial_guess` equivalent, 44-68 ms, ledger 211(ii)). It runs a third skimage seed when h_T wins (`h0=None`). The realistic total is about 190-230 ms per routed point, roughly 7-10 IC-GN iterations rather than five.

Fix: Correct the D22.15 expectation and list the components. Optionally note that the duplicate preprocessing is the price of keeping `fit_pattern`'s signature frozen.

### F09 (minor) -- requirements_D22.md D22.1 ('"always" ... no retry'), D22.8 ('"always" runs no retry (every point already had the FM row)')

Issue: The stated reason is false wherever the acceptance check kept h_T: the FM row was evaluated but never fitted. A routed point whose h_T won and whose fit failed is retried from its forced FM row under `"auto"` but not under `"always"`. `"always"` therefore converts fewer points than `"auto"`, contrary to its name and to its role as the xmap-free mode in V10(f) and the parity arms.

Fix: Either run the one retry under `"always"` as well, restricted to non-converged points whose first fit was seeded by h_T, or say explicitly in D22.1, D22.8 and the docstring that `"always"` means acceptance on every point with no retry and can convert fewer points than `"auto"`.

### F10 (minor) -- requirements_D22.md D22.5 ('a forced slot whose FM estimate fails is ... not refitted'), D22.8; validation_V10.md V10(i)

Issue: On the device the seam returns h_T for a forced slot whose FM estimate failed, and `run_lockstep` fits every finite real row in the same device batch. The runner reads `fourier_mellin_applied` only after the call, so the slot IS refitted from h_T. The result is inert, because it is deterministic and does not converge, but it contradicts the 'not refitted' clause and V10(i)'s no-second-fit spy. Only the CPU route, where the P = 1 seam call precedes the fit, can honour the clause as written.

Fix: Specify the device mechanism: route-2 slots with applied False are treated as non-real (inactive, like padded slots) in the retry batch. Alternatively, state 'fitted and discarded; the replacement rule makes it inert' and scope the V10(i) spy to the CPU route.

### F11 (minor) -- requirements_D22.md D22.3 step 1 and D22.4.1 (dead-band pixels included); validation_V10.md V10(b), (c)

Issue: The angle is computed from the reused spectrum of the whole bounding box, including the D4 dead-band cross. That cross, like camera row and column banding, is identical in reference and target and puts energy on the fx = 0 and fy = 0 axes. The Hann window removes edge effects, not interior lines, so the angle profile gains a fixed zero-lag component and is pulled toward 0. No V10 arm measures the angle with `dead_band` set; the only dead_band arm checks that the de-rotated crop contains those pixels.

Fix: Add a V10(b)/(c) arm on G1 with a constant dead-band cross in both images at 3 and 8 deg, measuring the bias. If the bias exceeds FM_ANGLE_TOL_DEG, either mask the look-up-table samples within one bin of the axes for reference and target alike and re-measure, or record the limitation in the docstring Notes and in F4.

### F12 (minor) -- validation_V10.md fixture G5 and V10(g)

Issue: G5's fixed-pattern noise is tuned 'until the (0, 0) premise holds' for the D5 seed. Noise strong enough to anchor the whitened phase correlation is also identical, and unrotated, in both FM spectra. It can dominate the radii 0.2-0.4 cycles/px, half of the radial mean, and pull theta_hat toward 0. De-rotating by a small angle does not break the anchor (ledger 207 saw escape only from about 0.4 deg). The V10(g) claims that the FM row is kept and converges where `"off"` does not can then fail for a fixture reason. On the real rim theta_hat over-read rather than locked, but G5 is synthetic.

Fix: Add a second run-time premise to G5: theta_hat within FM_ANGLE_TOL_DEG of the imposed twist. Use white per-pixel gain and offset noise rather than banded noise. If both premises cannot hold together, record that and drop the arm; the F8 anchor census covers the real-data case.

### F13 (minor) -- validation_V10.md fixture G8 and V10(i) ('the unrelated-pattern point keeps its first-pass result bitwise (fourier_mellin_seed 0, finite angle)'); plan_section12.md FM24

Issue: The arm needs a retried, non-converged point with a finite angle. Ledger 204(iv) says the unrelated pair 'fails by the D2.6 contract'. If that means NaN h, D22.8 excludes the point from the retry (finite h only), so its angle is NaN and FM24 has no killer. If instead the forced retry converges to a wrong optimum, FM24 survives as well.

Fix: Specify G8's unrelated point so its first pass is non-converged with a finite h, for example a large twist plus a projection-centre shift beyond the basin, or a small budget. Assert that premise at run time. Choose the retry budget so the retry provably cannot converge.

### F14 (minor) -- requirements_D22.md BLOCK 8 (amended D21.12 literal)

Issue: The new frozen message tells a user who passed seed_from_neighbors=True with backend='gpu' to use fourier_mellin='auto'. With seed_from_neighbors=True still set, that call raises the D22.11 ValueError.

Fix: Use: "seed_from_neighbors=True is not supported with backend='gpu'; use backend='cpu' for neighbour-seeded propagation, or seed_from_neighbors=False with fourier_mellin='auto' for large rotations about the detector normal".

### F15 (minor) -- requirements_D22.md D22.3 steps 5-6, D22.5

Issue: Four places are underspecified:
(a) 'ties to the LOWEST index' does not say in which order. In natural FFT order (lags 0..60 before -60..-1, as e2e.py does) a +k/-k tie picks +k; in lag order it picks -k.
(b) 'Lags wrapped to (-n/2, n/2]' differs from the prototype's [-n/2, n/2). Harmless inside +-60 bins, but it is a literal.
(c) 'From the unmasked neighbours' at the window edge should state that the raw c values outside the window are used.
(d) The acceptance compares a finite h_FM criterion against a possibly NaN h_T criterion with a finite h_T. 'Strictly lower' against NaN is False, so h_T is kept, and the intent is not stated. 'It equals fit_pattern(..., max_iterations=0)["residual"]' is not bitwise: the numpy twin's two-pass np.add.reduce differs from the BLAS dot, which matters for ties.

Fix: Pin each one explicitly. Recommended: ties to the lowest natural FFT index; lags in [-n/2, n/2); raw neighbour values; a non-finite h_T criterion keeps h_T, with the reason stated; 'equals to rounding', with ties decided on the namespace's own criterion values.

### F16 (minor) -- requirements_D22.md D22.7 (gate on the twist only); plan_section12.md F1, F2

Issue: Both the D5 seed's decorrelation and the FM estimate depend on the effective rotation of the crop: theta_eff ~= twist + (w1 xbar + w2 ybar)/(2 DD) (ledger 201; on the Si geometry theta_FM ~= w3 - 0.023 w1 + 0.218 w2). R_det already carries w1 and w2, so the gate could compute theta_eff for free. Gating on the twist alone ignores up to about 0.5 deg on the rim and more where w2 is larger.

Fix: Add theta_eff, computed with the bounding-box centroid, as an F1/F2 candidate gate quantity measured in the whole-map record. Alternatively, state why the twist alone is preferred.

### F17 (minor) -- validation_V10.md G7 and V10(h), (f)

Issue: No standing test links the gate's orientation convention to the actual projected pattern. G7 builds orientations with the gate's own algebra, `g_t = g_r (M^T R_det M)^T`. V10(f)'s ramp patterns come from the detector-frame deformation, not from those orientations. Routing uses |twist|, so a sign or frame-convention error is invisible. The only cross-check is the one-off probe of ledger 212(i), which lives in a scratchpad that will not survive the session.

Fix: Add a V10(h) arm that projects a pattern from a G7 g_t through the master-pattern projection (probe_frames.py's `project`) on both detectors. Assert that it equals the deformation construction (about 1e-10 relative), and that FM's theta_hat for that pair matches the gate twist within FM_ANGLE_TOL_DEG.

### F18 (minor) -- requirements_D22.md D22.18 (VRAM model gains fourier_mellin)

Issue: If `_default_batch_size(..., fourier_mellin=True)` lowers the default B below 32 on a smaller card, then P = min(32, B) drops. Every slot's FFT batch shape changes (D21.7.3), so on the GPU even unrouted points under `"auto"` stop matching `"off"` bitwise. That confounds the F3 'no point made worse' comparison.

Fix: Choose the default B from the `"off"` model and check FM's extra per-slot transient against the remaining headroom, halving only if it does not fit. Alternatively, record that FM can change B and therefore the bits of unrouted points, and compare F3 at a fixed explicit chunksize.

### F19 (minor) -- requirements_D22.md D22.3 step 3 (look-up table 'built once per reference'), D22.10 (FM state and numpy resident per routed reference, before the graph)

Issue: The look-up table depends only on (sr, sc) and the constants, so it is identical for every reference of a run. On the CPU route, each routed reference also holds a FourierMellinState, a numpy ReferenceResident and the bounding-box coordinates, about 15 MB at 480 px against 17.96 MB for the ReferenceState. All of these are built before the graph, which under `"always"` means every grain. None of it appears in the D16 information message.

Fix: Build one look-up table per run (cached by crop shape). Count the FM per-reference memory in the CPU information message, or build the numpy residents lazily per chunk.

### F20 (minor) -- requirements_D22.md D22.4.3; validation_V10.md ledger 210(iii) ('complete with d = t'); proto/e2e.py line 93 comment ('the composition mutant')

Issue: In Ernould 2021 eq. E.2 with the rotation centre at the PC (x0 = 0), delta = t, so Ernould's complete initialisation reads the lattice rotations from t. The draft's complete form uses d = R t and labels Ernould's own form a mutant. The two are valid orderings (Rz applied after the in-plane-axis rotations gives d = t; before gives d = R t) and differ at second order (24.47 against 24.52 px in ledger 210). This matters only if F5 is reopened.

Fix: In D22.4.3 and F5, state that both orderings are valid parameterisations and that Ernould uses d = t. Reserve the word 'mutant' for the partial row's T(t) R.

### F21 (minor) -- requirements_D22.md BLOCK 8 ('test_hrebsd_gpu.py:335 at f297867e')

Issue: At f297867e, SEED_FROM_NEIGHBORS_GPU_MESSAGE starts at line 337. Line 335 is the end of the backend message.

Fix: Cite test_hrebsd_gpu.py:337.

## critic B

Adversarial review of the Stage F spec drafts (requirements_D22.md, validation_V10.md, plan_section12.md, roadmap_stageF.md) through the conventions, integration and testability lens. I checked them against the frozen D21.5/D21.9.5 contract, the f297867e code and tests (copies in scratchpad/critic_ref/), the in-progress _batched.py/_gpu.py, and the prototype scripts and logs.

**What holds:**
- The keyword slot (after seed_from_neighbors) is valid. It keeps the Stage E slot pins true: test_hrebsd_gpu.py:2513/2532 check only backend's neighbours, and the engine order seed_from_neighbors, navigation_mask, backend, chunksize is preserved.
- The named freeze tests exist with the stated class names.
- The cited line numbers are correct (test_hrebsd_gpu.py:335/2975/5102, _engine.py:1212, _segmentation.py:224-231, ebsd.py:3492, CHANGELOG.rst:73-75).
- The "off" path can be bitwise by construction. The bibliography already carries ernould2020global and ernould2021integrated.
- Numbering and placement are right: D22 before "## Context", V10 after V9, plan section 12 after 11.5, roadmap Stage F before the NLPAR "---". The drafts are ASCII clean.

**Eight major problems:**
1. **No output channel.** Both runners return only packed (n, 12) rows; row_slots width 12 is frozen by the Stage E failing tests. So the FM angle and applied flag cannot reach the D22.9 props or the D22.8 "not refitted" rule.
2. **The acceptance needs the grain's subregion resident**, which the frozen seam never receives. The FourierMellinState field list omits it.
3. **GPU wiring is unspecified:** how the session builds the FM state, how the VRAM chooser is told FM is on, and the missing keyword on _vram_model_terms.
4. **V10(f) asserts a false outcome.** The spec gate's own sweep shows the FM seed converges the "unfittable" 8 deg point at the V8 PC (2 iterations, 0.053 px). Separately, the G8 fixture's twist ("3 to 4 deg") includes 3.0 deg, where the default seed converges at the V8 PC, so the premise fails.
5. **Headline evidence comes from other recipes.** The 113/113 sweep and the real -6 to 10 deg runs used the second prototype's whitened per-radius recipe or the method-study recipe, not the frozen D22.3 recipe. The frozen recipe is measured end to end only to 15 deg, yet the public docstring would claim "at least 30 deg".
6. **No frozen patch points** for the planted-failure, peak-search and gate-boundary tests.
7. **No test of the public method:** nothing checks that EBSD.hrebsd_dic forwards the keyword or copies the props, so both mutants survive.
8. **Missing dated amendments:** the Scope bullet, plan open question 5, D21.1's frozen check order, a D16 note, the V9(b) text, and a public tutorial bullet in hrebsd_dic.ipynb that becomes false.

There are also 16 minor convention and testability findings.

### C-F-MAJ-1 (major) -- requirements_D22.md D22.6 (output keys, 'The runner reads both after the call'), D22.8 ('a forced slot whose FM estimate fails ... is not refitted'), D22.9 (props), D22.11(2) (seed_extras is input only); plan_section12.md item 2

Issue: Neither runner has a way to return the FM outputs.
- CPU: _fit_chunk/_run_chunks return only the packed (n, _ROW_WIDTH=12) rows (`_engine.py:135`, `:1534`, blockwise new_axes j=12).
- GPU: `_run_chunks_gpu(..., row_slots)` has its argument list frozen by the Stage E failing-tests docstring as `row_slots={"width": 12, ...}`, and `_fit_batch` returns `(B, width)` rows.

D22 defines an input channel (seed_extras) but no output channel for fourier_mellin_angle / fourier_mellin_applied. So the D22.9 props cannot be filled.

The D22.8 rule cannot be implemented either. On the GPU, `_fit_batch` runs `run_lockstep` right after the seam, so a forced slot whose FM estimate failed (h_T returned) IS fitted again from h_T. The engine then cannot tell that row from a genuine retry result. On the CPU route the chunk has no way to say 'not refitted, keep the first result'.

Fix: Freeze the return path in D22.6, D22.9 and D22.11, plus a dated D21.9 note.
- On FM-on runs only, the packed row gains two slots, e.g. `row_slots` adds `"fm_angle": 12, "fm_seed": 13` with width 14, and the CPU chunk uses a 14-wide row. The runner writes them from the output keys. FM-off stays width 12, bitwise (the V9 pins).
- A forced slot whose `applied` is False is made inactive by the runner, like a padded slot: it is never fitted and gets a sentinel code. run_hrebsd_dic skips such rows when it replaces retry results.
- Add V10(i)/(k) arms for the packed width and the sentinel.
- Add a mutant: a forced-but-not-applied slot fitted from h_T.

### C-F-MAJ-2 (major) -- requirements_D22.md D22.5 ('through kernels.gather on the grain's subregion resident and kernels.final_criterion'), D22.6 (FourierMellinState field list), D22.10

Issue: The frozen seam cannot reach the subregion resident the acceptance needs.
- `seed_homographies(ctx, batch, target_spectra, seed_state)` sees only SeedState: bounds, reference_spectrum, precision, upsample_factor.
- The session passes only `seed_state` to the seam: `_gpu.py` `residents()` returns `(resident, seed_state)` and `_fit_batch` calls the seam with `seed_state` alone.
- The listed FourierMellinState fields (LUT, reference profile, box resident, n_theta, rho_min/max, search_deg) do NOT include the grain's subregion ReferenceResident.
- They also do not include the K shifts that `final_criterion(resident, values, shifts)` needs (`_batched.initial_shifts(ctx, resident, batch)`).

As specified, the frozen acceptance cannot be computed inside the seam. The only routes are to break the frozen signature, or to build a second subregion resident per reference. The second route doubles the r term against R_MAX = 2.

Fix: In D22.6, freeze a FourierMellinState field (e.g. `resident`) holding a REFERENCE to the grain's existing ReferenceResident. It must be the same object run_lockstep uses: no second upload, and it is evicted with its SeedState under R_MAX.
- State that the acceptance computes K the `initial_shifts` way on the slot targets.
- On the CPU route, the same object is built once per routed reference.
- Add a V10(k) identity spy: the resident used by the acceptance is the lockstep's resident.

### C-F-MAJ-3 (major) -- requirements_D22.md D22.6, D22.11(2), D22.18; Block 7 (D21.9.5 amendment)

Issue: The GPU wiring for FM is unspecified.
- The device runner learns about FM only through `seed_extras`. Nothing says how `_GpuSession.residents()` attaches `build_fourier_mellin_state`; that method calls `_batched.build_seed_state(ctx, state, *, precision, upsample_factor)` with a frozen argument list.
- Nothing says whether the FM state is built for grains with no routed slot (a 3.8 MB LUT plus the box resident per reference under "auto").
- Nothing says when `_default_batch_size(..., fourier_mellin=)` gets True. If "auto" with no pre-fit routes passes True, then "auto" and "off" can choose different default B on the same card.
- `_vram_model_terms(n_pixels, device_precision, seed_precision) -> (g, p, r)` (frozen by the Stage E failing-tests docstring; V9(o) calibrates g, p and r SEPARATELY) does not get the keyword. The FM p and r terms therefore have no function to live in or to be calibrated through.
- V10 maps D22.18 to the gated suite only.

Fix: Amend D22.11 and D22.18:
- FM is on in the device runner iff `seed_extras` carries "fourier_mellin_route".
- The FM state is built lazily in `residents()` the first time a sub-batch of that grain has a nonzero route. It is attached to the SeedState and evicted with it.
- `_vram_model_terms` also gains the keyword-only `fourier_mellin=False`.
- The chooser receives True iff any route flag is nonzero.
- Add a default-suite arm (V9(o) style): with `fourier_mellin=False`, all three functions return the Stage E value bitwise, and True >= False.

### C-F-MAJ-4 (major) -- validation_V10.md (f) and fixture G8; plan_section12.md task 1 (premises)

Issue: (f) asserts that 'the 8 deg out-of-frame point and the constant failed pattern keep their "off" outcomes'. That is false for the 8 deg point.
- The ramp map's RAMP_UNFITTABLE point (`test_hrebsd_seeding.py`, 8.0 deg in-plane at the V8 PC (0.49, 0.51, 0.5049)) is captured by FM in the spec gate's own run: `proto/out/sweep.log`, B_PCSEED_b05 inplane 8: 'fm True 2 0.0531'. Ledger 215's 6.62 deg in-frame angle is an accuracy limit, not a capture limit.
- Under "always", and under "auto" (twist 8 >= 1.5), the test would be red on a correct implementation or invite a wrong fix.
- 'fourier_mellin_seed is 1 on the routed ramp points' also fails under "always" for the reference point: an h_T criterion of about 1e-16 ties, so h_T is kept and the code is 0.
- The 9 deg out-of-plane rescue point has no stated expectation.

G8 problem: at the V8 PC the default seed CONVERGES at 3.0 deg (sweep.log B 3.0: 'tr True 14') and fails at 2.5 and 4.0 deg. A G8 twist anywhere in '3 to 4 deg' can break the stated premise.

Fix: Rewrite (f):
- The 8 deg point CONVERGES under both FM modes (code 1, error within a ledger 215 mirror-boundary band, MTP).
- The constant pattern keeps the D2.6 contract.
- Give the rescue point's expectation.
- List the expected fourier_mellin_seed per point, with reference points at 0.

Pin G8's twist at 4.0 deg and name its budget (MAX_ITERATIONS = 200). Keep the premise asserted at run time.

### C-F-MAJ-5 (major) -- requirements_D22.md D22 intro (lines 24-28), D22.16 (docstring capture sentence); roadmap_stageF.md paragraph; validation_V10.md ledger 202(ii)/203(ii)

Issue: The headline capture evidence was measured with recipes other than the frozen D22.3 recipe.
- 'A Fourier-Mellin (FM) seed captured every case of a 113-case synthetic sweep up to 30 deg' and 'every rigidly rotated real case from -6 to 10 deg' come from the second prototype's whitened per-radius recipe (`proto/fm.py` DEFAULT: method pc1d, rho 0.02-0.25, 64 radii; used by sweep.py and realrot.py) and from the method study.
- The frozen recipe is V8 (radial-mean ZNCC, rho 0.02-0.40). It was run end to end only on ledger 214's oracle cases (pure twists <= 5 deg plus combined rotations) and ledger 210(iii)'s four PC-shift cases (<= 15 deg). It was measured angle-only up to 19.13 deg (ledger 213).
- D22.16's public sentence 'the seed captures at least 30 deg' therefore has no measurement for the shipped recipe. 'At least' is also impossible past the +-30 deg search window.

Other record problems:
- Ledger 202(ii) says '1 to 8 iterations', then 'at most 9 with tilts'; sweep.log shows 9.
- The roadmap's 'fails ... from about 2-3 deg on rigidly rotated real Si' omits the default band-pass qualifier. Under the production `(None, None)` the translation seed reached 6 deg (ledger 203(ii)), above the map's largest input twist of 2.6 deg (ledger 212).

Fix: Label every capture number with its recipe (pc1d prototype, method study, or V8).
- In the D22 intro and the roadmap, state that V8's end-to-end capture is measured to 15 deg, and that capture beyond that is MTP via the weekly V10(e) arms.
- Reword D22.16 to 'up to the 30 degree search window', the figure filled at the implementation gate.
- Change '1 to 8' to '1 to 9'.
- Add the band-pass qualifier and the `(None, None)` 6 deg figure to the roadmap paragraph and the D22 intro, and state what that means for the real-data motivation.

### C-F-MAJ-6 (major) -- requirements_D22.md D22.6 (frozen names); validation_V10.md (b) planted correlation, (d) planted failures, (g) planted wrong angle and tie, (h) routing boundary, (k) 'never calls fourier_mellin_angles (spy)'

Issue: D22.6 freezes only FourierMellinState, build_fourier_mellin_state, fourier_mellin_angles, the gate function and the constants. The V10 arms need more patch points than that:
- patch 'the angle stage' and 'a translation stage returning a non-finite shift';
- patch 'the FM stage ... to return h_T';
- unit-test the peak search 'on a planted correlation', which takes a correlation, not spectra;
- route at `|twist| == FM_GATE_DEG` against the next float below it. That cannot be constructed from orientation matrices in float64, so the twist must be planted.

None of these has a frozen name or argument list, or a call-time module-global lookup rule (the D21.5(ii) and Stage E 'call-time seams' precedent). The failing tests have no stable targets, and an implementation that inlines these steps or binds them to locals makes the arms unpatchable.

Fix: Freeze in D22.6 (or require the failing-tests commit to record them in the module docstring, as D21.9.5 did) names and argument lists for:
- the peak search, e.g. `fourier_mellin_peak(xp, correlation, search_deg) -> (theta_deg, peak)`;
- the FM row builder, e.g. `fourier_mellin_rows(ctx, batch, target_spectra, seed_state, h_t, route) -> (rows, angle, applied)`;
- the de-rotation and translation step;
- the routing rule, e.g. `fourier_mellin_routes(twist_deg, gate_deg) -> int8`.

Require that seed_homographies, the runner and run_hrebsd_dic reach each of them, and twist_about_detector_normal, through the `_fourier_mellin` module global at call time.

### C-F-MAJ-7 (major) -- validation_V10.md (a), (e)-(j); requirements_D22.md D22.9; plan_section12.md item 4 (FM1-FM40)

Issue: Every FM-on oracle runs through the engine entry.
- No arm asserts that `EBSD.hrebsd_dic(fourier_mellin=...)` forwards the keyword. ebsd.py builds the engine call keyword by keyword (f297867e `ebsd.py:3460` onward).
- No arm asserts that the prop loop `for name in (*STAGE_A_PROP_NAMES, SEED_ROUND_PROP_NAME)` (`ebsd.py:3492`) copies the two new props into the returned CrystalMap.

A public method that drops the keyword silently runs "off", which is bitwise the default, so every "off" pin still passes. An unextended prop loop drops both props. Neither mutant is in FM1-FM40, and both survive V10.

Fix: Add to V10(a):
- a forwarding spy for all three values (the Stage E `test_public_method_forwards_backend` precedent, `test_hrebsd_gpu.py:2537`);
- an end-to-end public-method arm on the Ni map under "always", asserting both props on the returned CrystalMap with the D22.9 dtypes and shapes, and their absence under "off".

Add mutants FM41 (keyword not forwarded) and FM42 (prop loop not extended), each killed by these arms.

### C-F-MAJ-8 (major) -- requirements_D22.md amendment Blocks 2-9; plan_section12.md 'Spec files' paragraph and files-touched list; D22.16

Issue: The amendment set misses anchors that become false once D22 lands:
- the requirements Scope bullet (`requirements.md:115-119`: 'Fourier-Mellin ... deferred ... (not commissioned)');
- plan open question 5 (`plan.md:348-358`: 'Not yet commissioned; the default above stays in force until it is');
- D21.1's 'Order of checks, frozen' (`requirements.md:2023-2034`), into which D22.1 inserts three checks with no dated D21.1 amendment;
- the D16 determinism pledge, which D22.13 extends without the dated D16 note that D21.7 added;
- V9(b)'s prose (`validation.md:4673-4676`: 'so backend='cpu' and Fourier-Mellin are pinned as written'), which the amended literal invalidates;
- the PUBLIC tutorial `doc/tutorials/hrebsd_dic.ipynb:1877-1881` markdown bullet 'A Fourier-Mellin pre-rotation stage would lift it and is not built', which becomes false (D22.16 edits only hrebsd_si_indent.ipynb);
- the hrebsd_dic docstring Limitations paragraph ('The initial guess is a translation only phase cross-correlation ...', f297867e `ebsd.py:3343`).

Fix: Add dated blocks:
- Scope: 'SPECIFIED 2026-10-07 as Stage F, D22; opt-in'.
- Plan open question 5: 'SPECIFIED 2026-10-07: section 12; default unchanged (off)'.
- D21.1: 'AMENDED 2026-10-07 (D22.1): after the backend string check come the three FM checks, then the gpu-only checks'.
- A D16 dated note.
- A V9(b) dated note.

Extend D22.16 and the plan's files-touched list with one markdown edit of that hrebsd_dic.ipynb bullet (no stored output touched, nbval re-run once) and the docstring Limitations rewrite, pinned by the docstring test.

### C-F-MIN-1 (minor) -- requirements_D22.md D22.6 (output keys and the host skip); validation_V10.md (d)

Issue: D22.6 contradicts itself, and V10(d), on when the output keys exist.
- D22.6 says seed_homographies 'writes nothing' when the route key is absent OR all zero, and also that outputs are written 'only when the input key is present'. V10(d) says they are 'written only when the input key is present'. So with a present, all-zero key it is unclear whether outputs are written.
- 'NaN elsewhere / where an angle was estimated' conflicts with the masked full-P rule, under which an angle is computed on every slot.
- A planted seed_homographies (the V9(e) arms) writes no outputs at all, and the runner's behaviour then is undefined.

Fix: State one rule.
- Outputs exist iff the route key is present AND has a nonzero entry.
- The runner treats absent outputs as angle NaN and applied False on every slot.
- The angle is NaN on route-0 and padded slots even though it was computed.
- Add a V10(k) arm with an all-zero route key and one with a planted seam.

### C-F-MIN-2 (minor) -- validation_V10.md (g) criterion spy

Issue: 'A spy shows the criterion evaluated ... twice per routed slot and never on an unrouted one' contradicts D22.6. There the FM branch runs at the FULL sub-batch P, with unrouted slots computed and discarded, so a batched-kernel spy sees every slot.

The counted calls are also ambiguous: kernels.gather is called for the de-rotation as well as for the two criteria.

Fix: Restrict the exact-count arm to the CPU route at P = 1: final_criterion is called exactly twice per route-1 point and never for route 0 or route 2. For the numpy session, assert that criterion calls happen only in sub-batches with a nonzero route, at full P. Name which kernel's calls are counted.

### C-F-MIN-3 (minor) -- requirements_D22.md D22.5 ('on the CPU route it equals fit_pattern(state, target, h0=row, max_iterations=0)["residual"]')

Issue: The CPU route evaluates the criterion through the numpy twin's two-pass final_criterion. fit_pattern computes `residuals @ residuals` (a BLAS dot). They differ at about 2.5e-16 relative (ledger 103(iii): NUMPY_PARITY_RESIDUAL_RTOL = 5e-16), so 'equals' is false. A test that re-derives near-tie decisions through fit_pattern can flake.

Fix: Reword to 'agrees within NUMPY_PARITY_RESIDUAL_RTOL'. Require tests to read the decision from the 'applied' output, never to re-derive it through fit_pattern.

### C-F-MIN-4 (minor) -- requirements_D22.md D22.8 (GPU retry 'a new session'), D22.13

Issue: A new session re-runs `_default_batch_size` (with fourier_mellin=True), so the retry pass can run at a different B than the first pass, and at a different P below 32.
- D21.10.4's 'a completed run's results come wholly from its final B' and tech-stack (f)'s 'at a fixed ... batch size' then describe a two-B run.
- An out-of-memory halving inside the retry pass is unspecified.

Fix: State that the retry reuses the first pass's final B, passed as an explicit chunksize. An out-of-memory in the retry follows D21.10.4 for the retry subset only, and is recorded. Add this to 12.4 item 9.

### C-F-MIN-5 (minor) -- requirements_D22.md D22.8 ('whose first-pass h is finite (a D2.6 failure is not retried: its crop is unusable)')

Issue: The rationale is overclaimed. D2.6's NaN contract also arises from mid-fit exceptions (a non-finite step, a singular update, M[2,2] == 0), which is exactly where a fit from a spurious translation seed can diverge. It is not only an unusable crop.

Fix: Either retry NaN-h points too (an unusable crop makes the FM estimate fail, so the forced slot is not applied and not refitted, which already handles it), or reword the exclusion as a recorded choice ('a NaN-h point carries no last iterate'). List it under F7.

### C-F-MIN-6 (minor) -- validation_V10.md intro and fixtures; plan_section12.md item 1 and 12.4 item 16

Issue: Only the cupy_gpu fixture and the fake-cupy helpers are listed as duplicated. Under `--import-mode=importlib` many more helpers must be copied:
- (a), (j) and (k) use F6, the Ni map, `install_numpy_session`, `run_gpu_numpy`, `assert_properties_bitwise`, `seed_case` and `numpy_seed_context` (`test_hrebsd_gpu.py:825-1600`);
- G1 needs the deformed-master oracle helpers of test_hrebsd_engine.py;
- G2 needs `ramp_map` and `in_plane_fe` (`test_hrebsd_seeding.py:713-845`).

The CI estimate 'at most about 30 patterns at 480 px' is low:
- (f) alone fits the 10-point ramp map three times, with about 5 capped 200-iteration fits on the "off" premise;
- (e)'s starred arms are 5 x 2 runs;
- (j) runs the map about 6 times.

Fix: List every duplicated helper with its source line range and a 'copies kept identical' rule. Alternatively, revisit 12.4 item 16 and put the Stage F classes in test_hrebsd_gpu.py, avoiding roughly 800 lines of copies. Cache the expensive runs with lru_cache as V8 does. Give a per-class fit count and record the wall time.

### C-F-MIN-7 (minor) -- validation_V10.md intro ('four edits ... the docstring test in test_ebsd_hrebsd_dic.py'); plan_section12.md item 1

Issue: V10 counts an edit of 'the docstring test in test_ebsd_hrebsd_dic.py' among the existing pins. Plan item 1 instead puts 'the docstring test' in the new TestFourierMellinSwitch and says 'the hrebsd_dic docstring test gains the keyword'. None of the existing docstring tests needs an edit: `test_docstring_carries_the_documented_limitations`, `test_docstring_lists_every_stage_a_property`, `test_the_docstring_documents_the_keyword`, or Stage E's `test_the_docstring_documents_the_backend`.

Fix: Name exactly one test: a NEW `TestFourierMellinSwitch::test_the_docstring_documents_fourier_mellin` (the D21.17 precedent). Make the list three existing-pin edits, or name the edited test and say why it changes.

### C-F-MIN-8 (minor) -- plan_section12.md 12.2 (F1-F14) vs validation_V10.md (a) 'F6 (the four-point pre-Stage-D pin map)', G3 'the F3s geometry of V9'

Issue: The Stage F open questions F1-F14 collide with the V9 fixture names F1-F7, which V10 itself cites. For example, F6 is a fixture in V10(a) and 'The acceptance' in 12.2; F3 is 'The default value'. This invites mis-citation in ledger entries and test comments.

Fix: Either rename the open questions (e.g. FQ1-FQ14) or always cite fixtures as 'V9 F6'. Apply it consistently across D22, V10, plan 12 and the roadmap.

### C-F-MIN-9 (minor) -- requirements_D22.md D22.7 (all-zero UserWarning); validation_V10.md (h) '(literal)'

Issue: D22.7 gives no literal for the warning, but V10(h) asserts the literal.

The trigger 'a twist of exactly 0' is a float-equality test on atan2 of `M R_s M^T`. For identity orientations, M M^T is a Gram form and is exactly symmetric, so the twist is exactly 0. For a constant NON-identity placeholder (every point at the reference's orientation), `(M R_s) M^T` is not a Gram form, its antisymmetric part rounds to about 1e-17, and the warning does not fire.

Fix: Freeze the warning text in D22.7 (no stage letters). Define the trigger on the input: every fitted non-reference point's best rotation equals its reference's (equal quaternion data), or |twist| < 1e-9 deg. Add a constant non-identity arm to G7.

### C-F-MIN-10 (minor) -- requirements_D22.md D22.3 steps 3, 5, 6; validation_V10.md (b) LUT count pins

Issue: The text and the evidence disagree on details the exact pins depend on.
- D22.3.3 freezes 'a step of 1/min(sr, sc)' with 165/175 radii (the cost_v8.py arange rule). The accuracy evidence (h2h V8 and e2e) used `np.linspace(0.02, 0.40, round(0.38*min)+1)` (`proto/h2h.py` n_radii_of): 176 radii at 460x560. So the 252000-entry LUT pin describes a grid never measured for accuracy.
- 'Ties to the LOWEST index' does not say lowest FFT-output index (lag +k first) or lowest signed lag.
- 'Lags wrapped to (-n/2, n/2]' differs from the prototype's [-n/2, n/2).

Fix: Write the exact formulas:
- `rho_k = FM_RHO_MIN + k / min(sr, sc)` for `k = 0 .. floor((FM_RHO_MAX - FM_RHO_MIN) * min(sr, sc))`;
- the tie rule as the lowest index of the length-n FFT output order (lag 0, 1, ..., n/2, then -n/2+1, ..., -1), or another stated order;
- one stated lag range.

Note that the evidence grid differs (believed immaterial, re-measured at the gate).

### C-F-MIN-11 (minor) -- plan_section12.md item 4 (FM3, FM29; missing mutants); validation_V10.md (b), (l)

Issue: Several killers are weak or missing.
- FM3's 'ONLY killer' is the G3 band, pinned below the 0.18 deg bin-unit bias at 8 deg. V8's synthetic error reaches 0.123 deg (ledger 213) and V10 pins at about 2x margin, while V8's error on G3 is unmeasured (ledger 208 used real Si and the method-study recipe). The kill may have no margin.
- FM29's device killer, '(l) B and routing invariance', has no routing-invariance arm in (l).
- No mutant for de-rotating about the detector centre or the target's own PC. This is separable like FM2: the error is 2 sin(theta/2)|c|.
- No mutant for routing every CPU point through the numpy seam. It is equivalent on fixtures where the numpy seam equals skimage, so only a call-count spy kills it.
- The D/G suite tags promised in the item 4 intro are missing on most entries.

Fix: - Make a LUT-coordinate unit pin the primary FM3 killer: the bilinear weights reconstruct `(rho cos(theta) * sc, rho sin(theta) * sr)`. Keep G3 as the second.
- Add a gated arm to (l): a routed slot is bitwise the same with 1 of P and P of P slots routed.
- Add a rotation-centre mutant, killed by (d).
- Add a CPU over-routing mutant, killed by a `_batched.seed_spectra` call-count spy in (j).
- Tag every mutant D or G.

### C-F-MIN-12 (minor) -- requirements_D22.md D22 intro ('Johan's recorded notes ... (i) to (viii) ... are binding') and D22.10

Issue: Johan's note (viii) (`plan.md:1166-1175`) routes gated CPU points 'through the numpy seam ... (batched, then fitted with h0=row through the Stage D plumbing)'. D22.10 instead seeds at P = 1 inside the chunk function, with a new route block, and fits from h0 only when the FM row was applied. The draft records only the deviation from note (ii) (log-polar), and states the notes are binding and carried.

Fix: Record the (viii) deviation and its reason in D22.10: per-point independence from chunking by construction, and no host pre-pass over lazy patterns. Add it as a 12.4 item, with the batched host pre-pass through `_run_chunks(h0=...)` as the alternative.

### C-F-MIN-13 (minor) -- plan_section12.md files touched ('signals/ebsd.py (inside the fork-only hrebsd_dic method, its docstring and its prop loop only)')

Issue: The prop loop needs the new prop-name constants from `_engine`. They come through the module-level import block (`ebsd.py:59-63`). tech-stack.md's NLPAR fan-out names that block as an append-conflict zone with develop, so the files-touched statement is inaccurate.

Fix: State the edit in the merge-hygiene list: extend the existing `from kikuchipy.indexing._hrebsd._engine import (...)` block by one name (e.g. FOURIER_MELLIN_PROP_NAMES). Alternatively import inside the method.

### C-F-MIN-14 (minor) -- requirements_D22.md D22.10, D22.15, D22.18 (CPU half)

Issue: The CPU route builds, and holds for the whole run, the following for each reference with a routed point:
- a numpy SeedState;
- a FourierMellinState (3.8 MB LUT at 432x432, plus the box resident);
- a numpy ReferenceResident (about 6-8 float64 subregion planes, roughly 12-15 MB at 460x560).

Under "always" that is every grain. It roughly doubles the per-grain host precompute that D16's information message reports (`test_info_message_carries_the_memory_note`, `test_hrebsd_engine.py:2597`), on a machine whose commit memory is the binding constraint.

Fix: Add the FM per-reference host bytes to D22.15/D22.18 (CPU half). Extend the information message's memory note when FM is on. Either free a reference's numpy states after its grain's chunks, or record the bound.

### C-F-MIN-15 (minor) -- requirements_D22.md Block 5 (D15.6 amendment)

Issue: This departs from the D20.5 precedent. D20.5 amended D15.6 by reference, and seed_round was never inserted into D15.6's text. Inserting the FM props as a D15.6 sub-bullet leaves D15.6 listing the FM props but not seed_round.

Fix: Either follow D20.5 (no D15.6 text edit; D22.9 states that it amends D15.6 for FM runs), or add seed_round and both FM props in one dated D15.6 sub-bullet.

### C-F-MIN-16 (minor) -- plan_section12.md 12.2 F11

Issue: F11's 'Resolves' is an argument ('nothing inside a grain should exceed it'), not a measurement. It also ignores user-supplied grain_labels and tuple references, which allow misorientations above 5 deg.

Fix: Resolve F11 by the whole-map census in V10(m)(2): |theta_hat| and |twist| over the routed points. Reopen if any routed |twist| exceeds about 25 deg, or if any theta_hat sits at the window edge.
