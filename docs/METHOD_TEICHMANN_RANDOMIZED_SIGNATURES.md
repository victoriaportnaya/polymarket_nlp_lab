### Method based on “Signature Reconstruction from Randomized Signatures” 

#### 1) Data is a bounded-variation control path

- **Input path**: build a continuous (piecewise-linear) control path
  $$
  X:[0,T]\to\mathbb{R}^d
  $$
  from your streams (e.g. sentiment and market price). In our setting:
  $$
  X_t = (S_t,\;P_t) \in \mathbb{R}^2
  $$
- **Preprocessing**: resample to a fixed step size $\Delta t$ (forward fill / interpolation) so we can work with increments $\Delta X_k$.

As in the paper’s setup, continuous bounded-variation curves drive controlled differential equations (CDEs).

#### 2) Randomized signatures: features from a fixed CDE with random vector fields

The randomized signature are derived from solving a controlled ODE / CDE driven by the path $X$:

$$
Y_t = y + \sum_{i=1}^d \int_0^t V_i(Y_s)\, dX_s^i,
$$

where:
- $Y_t\in\mathbb{R}^N$ is the hidden state (the paper’s hidden dimension $N$),
- $V_i:\mathbb{R}^N\to\mathbb{R}^N$ are smooth vector fields with bounded derivatives,
- $y\in\mathbb{R}^N$ is an initial condition.

**Randomized signature construction:**
- Fix one draw of random vector fields $(V_i)_{i=1}^d$ (or a small ensemble of draws).
- Evaluate the map
  $$
  y \mapsto Y_T(y)
  $$
  for many initial values $y$ (a grid or random set).
- The collection of terminal states $\{Y_T(y)\}$ constitutes the randomized signature features of the input path.

The approach uses neural-network-type random vector fields, e.g.

$$
V_i(z) = \sigma(A_i z + b_i)
$$

and discusses deeper (depth-two) neural vector fields to expand the span of reconstructible signature terms.

#### 3) Signature reconstruction: linear regression on randomized-signature features

Let $\mathrm{Sig}^{\le m}(X)$ denote the truncated signature of $X$ up to order $m$ (iterated integrals up to depth $m$).

**Core reconstruction task:**
- Choose a signature coordinate (or all coordinates up to order $m$).
- Fit a *linear model* that maps randomized-signature features $\{Y_T(y)\}$ to the desired signature coordinates.

For each training path $X$ (or for sliding windows of one long path), we compute:
- Targets: selected signature coordinates (e.g. level-1 and level-2 signature terms).
- Inputs: randomized-signature feature vectors formed by concatenating $Y_T(y_j)$ across initial states $y_j$.

Then we solve a linear regression / ridge regression:
$$
\widehat{\mathrm{Sig}}^{\le m}(X) \approx \beta_0 + \beta^\top \Phi(X),
$$
where $\Phi(X)$ are the randomized-signature features.

Increasing hidden dimension $N$ (and using sufficiently expressive vector fields) increases how many signature terms can be reconstructed.

