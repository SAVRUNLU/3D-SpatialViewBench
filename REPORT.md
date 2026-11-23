# 3D-SpatialViewBench: Mental Rotation and Perspective-Taking in Text-Only LLMs

**Live Platform:** [streamlit.app](https://3d-spatialviewbench-9zz43nkaiop8du9zzbjzau.streamlit.app/) | **Repository:** [GitHub](https://github.com/SAVRUNLU/3D-SpatialViewBench)

## Summary

This report evaluates `deepseek-v4.1-flash`, `glm-5.3`, and `gemma-4-31b` on 3D-SpatialViewBench, a 60-item benchmark measuring a model's ability to compose rotation matrices and maintain a coordinate frame using only text-based symbolic computation. The harness made 180 total calls. All 180 calls completed with zero transport errors.

Against a 25% random guessing baseline, only `gemma-4-31b` performs above chance at 46.67%. The remaining models operate at the baseline floor: `deepseek-v4.1-flash` reaches 26.67%, and `glm-5.3` sits exactly at 25.00%. The data indicates a specific failure pattern where models compute the correct spatial vector during their reasoning process but bind it to the wrong option letter during final output formatting. 

## 1. Setup and Dataset Construction

Each item positions a camera at the world origin `(0,0,0)` and specifies a set of labeled objects at integer coordinates. The coordinate frame is right-handed: $+X$ is right, $+Y$ is up, and $-Z$ is forward. 

Models receive exact world coordinates and must compute the relative spatial vector between two objects in the camera's post-transformation frame. The final camera coordinate $\mathbf{v}_{cam}$ is calculated using the camera's orthonormal rotation basis $\mathbf{M}$ and its translation $\mathbf{t}$:

$$ \mathbf{v}_{cam} = \mathbf{M}^T (\mathbf{p} - \mathbf{t}) $$

The benchmark divides the mathematical operations into three distinct tiers:

| Level | Camera Operation | Required Math |
|---|---|---|
| 1 | Yaw rotation $\theta \in \{90^\circ, 180^\circ, 270^\circ\}$ | $\mathbf{M} = \mathbf{R}_y(\theta)$ |
| 2 | Translate $\mathbf{t}$, then Yaw $\theta$ | $\mathbf{M} = \mathbf{R}_y(\theta)$, offset by $-\mathbf{t}$ |
| 3 | Yaw $\theta$, then Pitch $\phi$ (local axis) | $\mathbf{M} = \mathbf{R}_y(\theta) \cdot \mathbf{R}_x(\phi)$ |

### Evaluation Protocol
The models were accessed via the OpenAI-compatible EVREN gateway. API calls were executed sequentially with `temperature=0` (greedy decoding) and a 1.5-second pause between requests. The system prompt forces the output into a strict JSON schema containing `"selected_option"` and `"reasoning"`. 

## 2. Quantitative Results

| Model | Accuracy | Level 1 | Level 2 | Level 3 | JSON Compliance | Latency (s) | Errors |
|---|---|---|---|---|---|---|---|
| gemma-4-31b | 46.67% | 60.0% | 25.0% | 55.0% | 96.67% | 44.7 | 0 |
| deepseek-v4.1-flash | 26.67% | 20.0% | 30.0% | 30.0% | 95.00% | 83.7 | 0 |
| glm-5.3 | 25.00% | 15.0% | 50.0% | 10.0% | 95.00% | 66.0 | 0 |

Statistical testing (two-tailed binomial, $p = 0.25$, $n = 60$) shows `gemma-4-31b` clears chance with $p \approx 0.0005$. `deepseek-v4.1-flash` ($p \approx 0.77$) and `glm-5.3` ($p \approx 1.0$) are indistinguishable from random option selection. 

The latency metrics separate reasoning effort from correct execution. `deepseek-v4.1-flash` averages 83.7 seconds per prompt but scores at the baseline floor. `gemma-4-31b` averages 44.7 seconds per prompt and operates nearly twice the baseline accuracy. The time spent computing does not correlate with spatial accuracy across the cohort.

## 3. Algorithmic Error Modes

Analysis of the incorrect model traces identifies two distinct failure states that are independent of basic arithmetic errors.

### 3.1 Derivation-Selection Dissociation
The dominant failure occurs when the model correctly derives the target direction word but selects a contradictory option letter. This disconnect affects both `glm-5.3` and `deepseek-v4.1-flash`.

An example from `glm-5.3` on item L3-002:
> "Camera-space components: forward component = dot with +Y = 3 (positive, forward). Right component = dot with right (−X) = −3 (negative, left). Up component = dot with up (−Z) = 2 (positive, up). So B is forward-left-up relative to D, which is option D."

The output payload for that call specifies `"selected_option": "A"`. Option A corresponds to `back-down-right`. The trace ends with a mathematically accurate calculation and binds the conclusion to the wrong string. 

### 3.2 Chirality Oscillation
When arithmetic fails, the failure typically traces back to the model changing its handedness convention mid-computation. The benchmark defines clockwise rotation against a right-handed frame, but models oscillate between coordinate rules (active vs passive rotation) during the reasoning chain.

This occurs on `gemma-4-31b` processing item L3-001:
> "New Forward vector becomes F' = (1,0,0)... Vector AC = (0,0,4)... AC dot Right(0,0,1) = 4. This suggests 'right'. Let me double check the yaw/pitch again. Yaw 90CCW: X->Z, Z->-X. No, standard rotation matrix for Y-axis: x'=x cos t + z sin t; z'=-x sin t + z cos t. For t=90deg: x'=z, z'=-x. So F(0,0,-1) becomes F'(-1,0,0)? No. If we rotate the coordinate system or the view?..."

The model determines the correct coordinate vector and then re-evaluates the rotation definition. It flips the yaw sign on the second pass and outputs the wrong option. 

## 4. Difficulty Scaling

Performance does not map cleanly to mathematical complexity. The accuracy curves across difficulty tiers are non-monotonic.

`gemma-4-31b` handles Level 1 and Level 3 at 60.0% and 55.0%, but fails at Level 2 with 25.0%. The Level 2 items require a camera translation ($\mathbf{t}$) prior to rotation. The traces show `gemma-4-31b` projecting the object position using the camera basis $\mathbf{M}^T$ but omitting the camera offset $-\mathbf{t}$, or applying the offset twice.

`glm-5.3` produces an inverted difficulty curve, scoring 50.0% on Level 2 while scoring 15.0% on Level 1 and 10.0% on Level 3. Option bias accounts for this anomaly. `glm-5.3` selects option A in 32 out of 60 items. When the correct answer is A, the model scores 10/15 (66.7%). On the remaining 45 items, it scores 5/45 (11.1%). The elevated Level 2 accuracy relies on option A appearing disproportionately in that specific subset rather than accurate translation calculation.

`deepseek-v4.1-flash` maintains a flat performance distribution across all levels (20.0% / 30.0% / 30.0%), rewriting the coordinate convention repeatedly without achieving higher resolution on the simpler operations.
