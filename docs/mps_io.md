# MPS & QPS Reader and Writer Architecture

This document describes the format specification, parsing algorithms, numerical choices, and limitations of the MPS / QPS input-output engine (`/core/io`).

## 1. Format Specification

The Mathematical Programming System (MPS) format is the industry-standard file representation for linear and mixed-integer programming problems. The Quadratic Programming System (QPS) extension adds a section for quadratic objective terms.

### Supported Sections
1. `NAME`: Problem name identifier.
2. `OBJSENSE` (Extension): `MIN` or `MAX` (default: `MIN`).
3. `ROWS`:
   - `N`: Objective or unconstrained row.
   - `G`: Greater-than-or-equal inequality: $A_i x \ge b_i$.
   - `L`: Less-than-or-equal inequality: $A_i x \le b_i$.
   - `E`: Equality constraint: $A_i x = b_i$.
4. `COLUMNS`:
   - Non-zero matrix coefficients $A_{ij}$ and linear objective coefficients $c_j$.
   - Integer variable markers:
     - `'MARKER'` `'INTORG'`: Begins integer variable block.
     - `'MARKER'` `'INTEND'`: Ends integer variable block.
5. `RHS`: Right-hand side values $b_i$.
6. `RANGES`:
   - Converts one-sided inequalities into boxed constraints:
     - For row $i$ of type `G`: $b_i \le A_i x \le b_i + |r_i|$.
     - For row $i$ of type `L`: $b_i - |r_i| \le A_i x \le b_i$.
     - For row $i$ of type `E` with $r_i \ge 0$: $b_i \le A_i x \le b_i + r_i$.
     - For row $i$ of type `E` with $r_i < 0$: $b_i + r_i \le A_i x \le b_i$.
7. `BOUNDS`:
   - `UP`: Upper bound ($x_j \le u_j$, default lower bound remains 0).
   - `LO`: Lower bound ($x_j \ge l_j$, default upper bound remains $+\infty$).
   - `FX`: Fixed variable ($x_j = val$).
   - `FR`: Free variable ($-\infty < x_j < +\infty$).
   - `MI`: Negative variable ($-\infty < x_j \le 0$).
   - `PL`: Positive variable ($0 \le x_j < +\infty$).
   - `BV`: Binary variable ($x_j \in \{0, 1\}$).
   - `UI`: Upper integer ($x_j \in \mathbb{Z}, x_j \le u_j$).
   - `LI`: Lower integer ($x_j \in \mathbb{Z}, x_j \ge l_j$).
8. `QUADOBJ` / `QMATRIX`:
   - Encodes symmetric quadratic objective matrix $Q$ representing $\frac{1}{2} x^T Q x$.
   - Format: `col1 col2 coefficient`.
9. `ENDATA`: File termination marker.

---

## 2. Parsing Algorithm

### Fixed vs. Free Format Handling
- **Fixed Format**: Detects whether line contents align with classic 6-field fixed columns:
  - Field 1: columns 2-3 (Indicator / Type)
  - Field 2: columns 5-12 (Name)
  - Field 3: columns 15-22 (Row / Column Name)
  - Field 4: columns 25-36 (Value 1)
  - Field 5: columns 40-47 (Row / Column Name)
  - Field 6: columns 50-61 (Value 2)
- **Free Format**: Splits tokens by whitespace while respecting single-quoted strings (e.g. `'MARKER'`).
- The parser automatically adapts to either format transparently.

### Two-Pass Construction
1. **Pass 1 (Headers & Rows)**:
   - Identifies problem name, sense, and declares all rows and objective rows.
2. **Pass 2 (Columns & Coefficients)**:
   - Reads non-zeros into coordinate triplets $(i, j, v)$.
   - Tracks integer ranges via `INTORG` / `INTEND` markers.
   - Registers variable names and ensures consistent column indexing.
3. **Pass 3 (RHS, RANGES, BOUNDS, Q)**:
   - Assigns lower/upper row limits and column limits.
   - Populates quadratic triplets.
4. **Assembly**:
   - Converts triplets into synchronized CSC and CSR sparse matrices via `SparseMatrix::from_triplets()`.

---

## 3. Serialization Algorithm (Writer)
- `write_mps(const Problem& problem, const std::string& filename)` exports clean, standard fixed/free MPS or QPS files.
- Preserves explicit row/column names, integer markers, ranges, and `QUADOBJ` blocks.

---

## 4. Known Limitations & Numerical Choices
- Values parsed as double-precision floats (`std::stod`).
- Variables default to $[0, +\infty)$ unless declared otherwise, conforming to the standard MPS specification.
- Zero coefficients are filtered out during triplet accumulation.
