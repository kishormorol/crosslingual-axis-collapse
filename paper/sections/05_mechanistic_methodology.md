# 4. Mechanistic Subspace Geometry and Steering Formulation

## 4.1 Formalizing Over-Refusal and Safe-Completion Subspaces

Consider a causal language model with $L$ transformer layers and hidden dimension $d$. For scenario $i \in \{1, \dots, N\}$, script $s \in \{\text{en}, \text{bn}, \text{bn\_latn}\}$, and intent $I \in \{\text{bg}, \text{bb}, \text{harm}\}$, let $\mathbf{h}_{i,s,I}^{(\ell)} \in \mathbb{R}^d$ denote the last-token residual stream activation at layer $\ell \in \{1, \dots, L\}$ formatted under the model's instruction chat template [3, 4]. Within per triplet $(i, s)$, the three intents isolate two distinct behavioral transitions:

$$\mathbf{d}_{\text{over}, i, s}^{(\ell)} = \mathbf{h}_{i, s, \text{bb}}^{(\ell)} - \mathbf{h}_{i, s, \text{bg}}^{(\ell)}, \qquad \mathbf{d}_{\text{safe}, i, s}^{(\ell)} = \mathbf{h}_{i, s, \text{bb}}^{(\ell)} - \mathbf{h}_{i, s, \text{harm}}^{(\ell)}$$

Here, $\mathbf{d}_{\text{over}, i, s}^{(\ell)}$ captures the **lexical sensitivity (over-refusal) shift** induced solely by introducing a sensitive domain keyword $k_i$ into a benign query, whereas $\mathbf{d}_{\text{safe}, i, s}^{(\ell)}$ captures the **safe-completion vs. harm boundary** separating a legitimate educational boundary query from a genuinely harmful request containing the same keyword $k_i$. Averaging across scenarios $i \in \mathcal{D}_{\text{train}}$ yields the script-specific **Over-Refusal Axis** $\mathbf{v}_{\text{over}, s}^{(\ell)}$ and **Safe-Completion Axis** $\mathbf{v}_{\text{safe}, s}^{(\ell)}$:

$$\mathbf{v}_{\text{over}, s}^{(\ell)} = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i \in \mathcal{D}_{\text{train}}} \mathbf{d}_{\text{over}, i, s}^{(\ell)}, \qquad \mathbf{v}_{\text{safe}, s}^{(\ell)} = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i \in \mathcal{D}_{\text{train}}} \mathbf{d}_{\text{safe}, i, s}^{(\ell)}$$

We additionally define the **True Refusal Axis** $\mathbf{v}_{\text{ref}, s}^{(\ell)} = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i} (\mathbf{h}_{i, s, \text{harm}}^{(\ell)} - \mathbf{h}_{i, s, \text{bg}}^{(\ell)})$. In a well-calibrated representation space, $\mathbf{v}_{\text{over}, s}^{(\ell)}$ and $\mathbf{v}_{\text{safe}, s}^{(\ell)}$ remain nearly orthogonal so that linear readouts can distinguish benign boundary queries from harmful requests without being confounded by surface keyword presence.

## 4.2 Defining the Cross-Lingual Axis Collapse Index (`CLACI`)

To quantify geometric interference between the over-refusal and safe-completion subspaces at layer $\ell$ in script $s$, we measure both the subspace cosine alignment and the principal angle $\theta_s^{(\ell)} = \arccos\big(|\cos(\mathbf{v}_{\text{over}, s}^{(\ell)}, \mathbf{v}_{\text{safe}, s}^{(\ell)})|\big) \in [0^\circ, 90^\circ]$. At the individual scenario level $i$, we define the **Cross-Lingual Axis Collapse Index (`CLACI`)** across the mid-to-late semantic layers $\mathcal{L}_{\text{mid-late}} = \{14, \dots, 22\}$:

$$\text{CLACI}_{i, s} = \frac{1}{|\mathcal{L}_{\text{mid-late}}|} \sum_{\ell \in \mathcal{L}_{\text{mid-late}}} \left| \frac{\langle \mathbf{d}_{\text{over}, i, s}^{(\ell)},\, \mathbf{d}_{\text{safe}, i, s}^{(\ell)} \rangle}{\|\mathbf{d}_{\text{over}, i, s}^{(\ell)}\|_2 \, \|\mathbf{d}_{\text{safe}, i, s}^{(\ell)}\|_2} \right| \in [0, 1]$$

A lower $\text{CLACI}_{i, s}$ (and larger principal angle $\theta_s^{(\ell)}$) indicates well-separated helpfulness and harmlessness axes, whereas a higher $\text{CLACI}_{i, s}$ (and smaller $\theta_s^{(\ell)}$) indicates that the model conflates benign boundary queries with refusal-triggering representations.

![Figure 2: Layer-Wise Principal Angle Trajectory Between Over-Refusal and Safe-Completion Subspaces](/Users/kishormorol/.gemini/antigravity-cli/brain/ecdf6b31-5c8a-4417-910b-e216edd16bd6/fig2_layer_trajectory.svg)

## 4.3 Cross-Lingual Orthogonal Safe-Completion Steering (`CL-OSCS`)

When $\text{CLACI}_{i, s}$ is elevated, standard Contrastive Activation Addition (`CAA`) [4] along $\mathbf{v}_{\text{safe}, s}^{(\ell)}$ inadvertently projects energy along the refusal subspace, either failing to rescue benign boundary prompts or destabilizing refusal on harmful prompts. Furthermore, non-English activations exhibit a large script-centroid offset $\boldsymbol{\mu}_s^{(\ell)} = \frac{1}{3|\mathcal{D}_{\text{train}}|} \sum_{i, I} \mathbf{h}_{i, s, I}^{(\ell)}$ relative to the English hub $\boldsymbol{\mu}_{\text{en}}^{(\ell)}$ [26, 31], and high subword fertility dilutes per-token steering norms across fragmented sequences.

We formulate **Cross-Lingual Orthogonal Safe-Completion Steering (`CL-OSCS`)** via three sequential operations at layer $\ell \in \mathcal{L}_{\text{mid-late}}$:

1. **Script-Centroid Mean-Centering**: We align script $s$ representations to the English semantic hub by removing the training-fold script shift $\boldsymbol{\delta}_s^{(\ell)} = \boldsymbol{\mu}_s^{(\ell)} - \boldsymbol{\mu}_{\text{en}}^{(\ell)}$:
   $$\tilde{\mathbf{h}}_{i, s}^{(\ell)} = \mathbf{h}_{i, s}^{(\ell)} - \beta \, \boldsymbol{\delta}_s^{(\ell)}$$
2. **Cross-Lingual Gram-Schmidt Orthogonal Projection**: Let $\hat{\mathbf{u}}_{\text{ref}, \text{en}}^{(\ell)} = \mathbf{v}_{\text{ref}, \text{en}}^{(\ell)} / \|\mathbf{v}_{\text{ref}, \text{en}}^{(\ell)}\|_2$ denote the unit English true-refusal direction, and let $\mathbf{v}_{\text{safe}, \text{en}}^{(\ell)}$ denote the English safe-completion direction. We project $\mathbf{v}_{\text{safe}, \text{en}}^{(\ell)}$ onto the orthogonal complement of $\hat{\mathbf{u}}_{\text{ref}, \text{en}}^{(\ell)}$ and normalize to unit length:
   $$\mathbf{v}_{\perp}^{(\ell)} = \mathbf{v}_{\text{safe}, \text{en}}^{(\ell)} - \langle \mathbf{v}_{\text{safe}, \text{en}}^{(\ell)},\, \hat{\mathbf{u}}_{\text{ref}, \text{en}}^{(\ell)} \rangle \, \hat{\mathbf{u}}_{\text{ref}, \text{en}}^{(\ell)}, \qquad \hat{\mathbf{v}}_{\perp}^{(\ell)} = \frac{\mathbf{v}_{\perp}^{(\ell)}}{\|\mathbf{v}_{\perp}^{(\ell)}\|_2}$$
   By construction, $\langle \hat{\mathbf{v}}_{\perp}^{(\ell)},\, \hat{\mathbf{u}}_{\text{ref}, \text{en}}^{(\ell)} \rangle = 0$, guaranteeing zero first-order interference with the core refusal axis.
3. **Inverse-Fertility Adaptive Scaling**: Given prompt subword fertility $F_\tau(x)$ and English reference fertility $\bar{F}_{\tau, \text{en}}$, we define the fertility-compensated steering coefficient $\gamma_\tau(x) = \alpha \cdot (F_\tau(x) / \bar{F}_{\tau, \text{en}})^{-\eta}$ (with $\eta = 0.25$) paired with a learned zero-sum contrastive gate $g_s^{(\ell)}(\tilde{\mathbf{h}}) \in [-1, 1]$ derived from the training-fold boundary-vs-harm separator:
   $$\mathbf{h}_{i, s, \text{CL-OSCS}}^{(\ell)} = \tilde{\mathbf{h}}_{i, s}^{(\ell)} + \gamma_\tau(x) \, g_s^{(\ell)}\big(\tilde{\mathbf{h}}_{i, s}^{(\ell)}\big) \, \hat{\mathbf{v}}_{\perp}^{(\ell)}$$
