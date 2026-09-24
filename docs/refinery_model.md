# Refinery Planning Model: Mathematical Formulation & Architecture

## 1. Executive Summary & Problem Scope

Refinery planning is the central optimization problem of downstream oil and gas operations. It determines:
1. **Crude Selection**: How much of each crude oil to purchase, accounting for spot prices, logistics limits, and quality characteristics (API gravity, sulfur percentage, distillation curves).
2. **Unit Operations**: Throughput and operating modes across distillation and conversion units (ADU, VDU, FCC, Hydrocracker, Reformer).
3. **Product Blending**: How intermediate streams are blended into commercial finished products (Gasoline, Diesel, Jet Fuel, Heavy Fuel Oil, LPG) meeting strict quality specifications (minimum Octane number, maximum Sulfur ppm, maximum Density) and contractual market demand.

The model is formulated as a **Linear Program (LP)** for continuous pooling and operations, and extended to a **Mixed-Integer Linear Program (MILP)** to model discrete unit on/off commitments, minimum operational turndown capacities, and fixed operating charges.

---

## 2. Mathematical Formulation

### 2.1 Sets & Indices
- $c \in \mathcal{C}$: Crude types (e.g. *Arab Light*, *Brent*, *Maya Heavy*, *Bonny Light*).
- $u \in \mathcal{U}$: Processing units:
  - $\text{ADU}$: Atmospheric Distillation Unit
  - $\text{VDU}$: Vacuum Distillation Unit
  - $\text{FCC}$: Fluid Catalytic Cracking Unit
  - $\text{HDT}$: Hydrotreating Unit
  - $\text{REF}$: Catalytic Reforming Unit
- $s \in \mathcal{S}$: Intermediate refinery streams (Naphtha, Kerosene, Gas Oil, Vacuum Gas Oil, Residue, Cracked Gas Oil, Reformate).
- $p \in \mathcal{P}$: Finished products (LPG, Gasoline Regular, Gasoline Premium, Jet Fuel, Ultra-Low Sulfur Diesel, Heavy Fuel Oil).
- $q \in \mathcal{Q}$: Quality attributes (Octane rating RON, Sulfur content wt%, API gravity, Cetane number).

### 2.2 Decision Variables
- $x_c \ge 0$: Volume of crude $c$ purchased and processed (barrels/day, bpd).
- $f_{u} \ge 0$: Total feed throughput processed in unit $u$ (bpd).
- $w_{s, p} \ge 0$: Volume of intermediate stream $s$ blended into product $p$ (bpd).
- $y_p \ge 0$: Total volume of finished product $p$ produced and sold (bpd).
- $z_u \in \{0, 1\}$ *(MILP only)*: Operational binary indicating whether unit $u$ is active ($z_u = 1$) or shut down ($z_u = 0$).

---

### 2.3 Constraints

#### 1. Crude Availability
Each crude has market purchasing limits:
$$0 \le x_c \le \text{MaxSupply}_c \quad \forall c \in \mathcal{C}$$

#### 2. Atmospheric Distillation Yields & Mass Balance
Crude distillation separates crude $c$ into primary distillation cuts according to assay yields $\gamma_{c, s}$:
$$\text{Production of stream } s = \sum_{c \in \mathcal{C}} \gamma_{c, s} x_c$$
Total ADU throughput equals total crude processed:
$$f_{\text{ADU}} = \sum_{c \in \mathcal{C}} x_c$$

#### 3. Unit Processing Capacities & Semi-Continuous Turndown (MILP)
In the LP model:
$$f_u \le \text{Cap}_u^{\max} \quad \forall u \in \mathcal{U}$$

In the MILP model, units require minimum turndown levels ($\text{Cap}_u^{\min}$) to avoid catalyst fouling or thermal instability:
$$\text{Cap}_u^{\min} \cdot z_u \le f_u \le \text{Cap}_u^{\max} \cdot z_u \quad \forall u \in \mathcal{U}, \; z_u \in \{0, 1\}$$

#### 4. Intermediate Stream Conversion & Conservation
Conversion units transform feed streams into higher-value intermediates:
$$f_u = \sum_{s \in \text{Feed}(u)} w_{s, u}$$
$$\text{Output of stream } s' = \sum_{u} \eta_{u, s'} f_u$$
Stream balances enforce that consumption does not exceed generation:
$$\sum_{u \text{ consuming } s} w_{s, u} + \sum_{p \in \mathcal{P}} w_{s, p} \le \text{Total Produced}_s$$

#### 5. Product Blending Conservation
$$\sum_{s \in \text{Components}(p)} w_{s, p} = y_p \quad \forall p \in \mathcal{P}$$

#### 6. Product Demand Bounds
$$D_p^{\min} \le y_p \le D_p^{\max} \quad \forall p \in \mathcal{P}$$

#### 7. Linear Quality Specifications
For properties that blend linearly by volume (e.g. Cetane index, Octane blending value, API gravity):
- Minimum specification:
  $$\sum_{s} q_{s, k} w_{s, p} \ge Q_{p, k}^{\min} y_p \implies \sum_{s} (q_{s, k} - Q_{p, k}^{\min}) w_{s, p} \ge 0$$
- Maximum specification (e.g. sulfur, benzene):
  $$\sum_{s} q_{s, k} w_{s, p} \le Q_{p, k}^{\max} y_p \implies \sum_{s} (q_{s, k} - Q_{p, k}^{\max}) w_{s, p} \le 0$$

---

### 2.4 Objective Function: Net Margin Maximization

The economic objective maximizes gross refining margin ($/day):
$$\max \quad \sum_{p \in \mathcal{P}} \text{Price}_p y_p - \sum_{c \in \mathcal{C}} \text{Cost}_c x_c - \sum_{u \in \mathcal{U}} \text{VOpCost}_u f_u - \sum_{u \in \mathcal{U}} \text{FixedCost}_u z_u$$
where:
- $\text{Price}_p$: Selling price of product $p$ ($/bbl).
- $\text{Cost}_c$: Acquisition and transport cost of crude $c$ ($/bbl).
- $\text{VOpCost}_u$: Variable operating cost per barrel (catalyst, utilities, steam, hydrogen).
- $\text{FixedCost}_u$: Daily fixed operating overhead incurred when unit $u$ is active.

---

## 3. Successive Linear Programming (SLP) for Nonlinear Quality Blending

Certain critical blend specifications (notably sulfur content with hydrotreated and non-hydrotreated pooling, or Reid Vapor Pressure) exhibit nonlinear pooling relationships:
$$\text{Actual Quality } Q_p = \frac{\sum_s q_s w_{s, p}}{\sum_s w_{s, p}}$$
When stream quality $q_s$ itself depends on upstream pool operating points, the constraint becomes bilinear:
$$\sum_s q_s(x) w_{s, p} \le Q_{p}^{\max} y_p$$

### 3.1 SLP Algorithm (Frank-Wolfe / Bilinear Linearization)
1. **Initialize**: Guess initial stream qualities $q_s^{(0)}$ based on nominal crude assay values.
2. **Solve LP**: Formulate and solve the linearized LP model using the indigenous LP solver to obtain flows $w_{s, p}^{(k)}$ and throughputs $x^{(k)}$.
3. **Evaluate Real Properties**: Compute true stream qualities $q_s(x^{(k)})$ using exact nonlinear physical relations.
4. **Linearize & Step**: Update linearized coefficients with trust-region damping:
   $$q_s^{(k+1)} = (1 - \alpha) q_s^{(k)} + \alpha q_s(x^{(k)})$$
5. **Check Convergence**: If $\max_s |q_s^{(k+1)} - q_s^{(k)}| \le \epsilon_{SLP}$, terminate with convergent blend.
