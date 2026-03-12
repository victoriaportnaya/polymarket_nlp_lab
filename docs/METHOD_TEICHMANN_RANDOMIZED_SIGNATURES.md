### Method (fully aligned to “Signature Reconstruction from Randomized Signatures”)

This section rewrites our approach to be **entirely based on** the randomized-signature / CDE reconstruction framework of Glückstad–Muça Cirone–Teichmann ([paper PDF](file:///Users/victoriaportnaya/Downloads/Teichmann.pdf)).

#### 1) Data as a bounded-variation control path

- **Input path**: build a continuous (piecewise-linear) control path
  $$
  X:[0,T]\to\mathbb{R}^d
  $$
  from your streams (e.g. sentiment and market price). In our setting:
  $$
  X_t = (S_t,\;P_t) \in \mathbb{R}^2
  $$
- **Preprocessing**: resample to a fixed step size $\Delta t$ (forward fill / interpolation) so we can work with increments $\Delta X_k$.

This matches the paper’s setup: continuous bounded-variation curves drive controlled differential equations (CDEs).

#### 2) Randomized signatures = features from a fixed CDE with random vector fields

The randomized signature is obtained by solving a **controlled ODE / CDE** driven by the path $X$:

$$
Y_t = y + \sum_{i=1}^d \int_0^t V_i(Y_s)\, dX_s^i,
$$

where:
- $Y_t\in\mathbb{R}^N$ is the hidden state (the paper’s **hidden dimension** $N$),
- $V_i:\mathbb{R}^N\to\mathbb{R}^N$ are smooth vector fields with bounded derivatives,
- $y\in\mathbb{R}^N$ is an initial condition.

**Randomized signature construction (paper’s central object):**
- Fix one draw of random vector fields $(V_i)_{i=1}^d$ (or a small ensemble of draws).
- Evaluate the map
  $$
  y \mapsto Y_T(y)
  $$
  for **many initial values** $y$ (a grid or random set).
- The collection of terminal states $\{Y_T(y)\}$ constitutes the **randomized signature features** of the input path.

The paper emphasizes neural-network-type random vector fields, e.g.

$$
V_i(z) = \sigma(A_i z + b_i)
$$

and discusses deeper (depth-two) neural vector fields to expand the span of reconstructible signature terms.

#### 3) Signature reconstruction = linear regression on randomized-signature features

Let $\mathrm{Sig}^{\le m}(X)$ denote the truncated signature of $X$ up to order $m$ (iterated integrals up to depth $m$).

**Core reconstruction task in the paper:**
- Choose a signature coordinate (or all coordinates up to order $m$).
- Fit a *linear model* that maps randomized-signature features $\{Y_T(y)\}$ to the desired signature coordinates.

Operationally, for each training path $X$ (or for sliding windows of one long path), we compute:
- Targets: selected signature coordinates (e.g. level-1 and level-2 signature terms).
- Inputs: randomized-signature feature vectors formed by concatenating $Y_T(y_j)$ across initial states $y_j$.

Then we solve a linear regression / ridge regression:
$$
\widehat{\mathrm{Sig}}^{\le m}(X) \approx \beta_0 + \beta^\top \Phi(X),
$$
where $\Phi(X)$ are the randomized-signature features.

Per the paper, increasing hidden dimension $N$ (and using sufficiently expressive vector fields) increases how many signature terms can be reconstructed.

#### 4) How this answers the Polymarket research questions (within the paper’s framework)

Everything below uses **only** operations allowed by the paper’s mechanism: build $X$, compute randomized-signature features via a CDE, and linearly reconstruct signature coordinates / functionals.

- **Nonlinear lags / lead–lag**: define $X_t=(S_t,P_{t-\tau})$ and reconstruct signature features on windows; compare reconstructed signature coordinates across $\tau$.
- **“Disagreement regimes”**: compare reconstructed level-2 antisymmetric term (signed area in 2D) between regime labels (agreement vs disagreement); this is a signature-coordinate view of the “area between streams”.
- **Model selection**: treat *predictive* tasks (e.g. forward return) as functionals of the path; use randomized-signature features $\Phi(X)$ directly as inputs to a linear model (or reconstruct signature then regress), consistent with the paper’s “linear combinations of CDE solutions approximate nonlinear path functionals” framing.

#### 5) Practical knobs (paper-consistent)

- **Hidden dimension $N$**: controls expressive capacity; paper discusses constraints/relations between $N$ and reconstructible depth/order.
- **Vector field family**: shallow vs depth-two neural vector fields; deeper fields help break algebraic relations and improve reconstructibility.
- **Number of initial states** $M$: more $y$ samples increases the linear span of available randomized-signature features.
- **Regularization**: ridge/lasso when $MN$ is large relative to training windows.

