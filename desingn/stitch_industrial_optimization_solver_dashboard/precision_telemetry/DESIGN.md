---
name: Precision Telemetry
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#3f4850'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#707881'
  outline-variant: '#bfc7d2'
  surface-tint: '#006398'
  primary: '#006194'
  on-primary: '#ffffff'
  primary-container: '#007bb9'
  on-primary-container: '#fdfcff'
  inverse-primary: '#93ccff'
  secondary: '#565e74'
  on-secondary: '#ffffff'
  secondary-container: '#dae2fd'
  on-secondary-container: '#5c647a'
  tertiary: '#006948'
  on-tertiary: '#ffffff'
  tertiary-container: '#00855d'
  on-tertiary-container: '#f5fff7'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#cce5ff'
  primary-fixed-dim: '#93ccff'
  on-primary-fixed: '#001d31'
  on-primary-fixed-variant: '#004b73'
  secondary-fixed: '#dae2fd'
  secondary-fixed-dim: '#bec6e0'
  on-secondary-fixed: '#131b2e'
  on-secondary-fixed-variant: '#3f465c'
  tertiary-fixed: '#85f8c4'
  tertiary-fixed-dim: '#68dba9'
  on-tertiary-fixed: '#002114'
  on-tertiary-fixed-variant: '#005137'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
  body-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  mono-metric-xl:
    fontFamily: JetBrains Mono
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.03em
  mono-metric-md:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  mono-data-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '400'
    lineHeight: 14px
  label-caps:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.04em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-desktop: 1.25rem
  margin: 1rem
  margin-desktop: 1.5rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style

This design system targets operations research engineers, computational scientists, and infrastructure architects running mission-critical mathematical optimization workloads. The visual language conveys rigorous mathematical authority, instant legibility under cognitive load, and absolute operational reliability. 

Drawing from modern high-density technical interfaces and flight-deck control-room instrumentation, the style marries utilitarian precision with refined digital craft. It avoids decorative fluff or exaggerated radii, favoring dense structured grids, disciplined visual hierarchies, and unambiguous semantic status mapping. Visual tension is resolved through crisp edge definitions, disciplined neutral planes, and deliberate micro-accents of electric cyan reserved strictly for interactive vectors, active states, and focal metrics.

## Colors

The palette establishes an ultra-clean, clinical operating environment optimized for long-duration viewing and dense telemetry scanning.

### Surface & Neutral Architecture
- **Canvas Base (`#F8FAFC` to `#F1F5F9`)**: Soft off-white and cool slate backdrops providing stable visual grounding without high-glare eye strain.
- **Surface Elevation 0 (`#FFFFFF`)**: Pure white reserved for cards, data grids, operational panels, and metric containers.
- **Structural Dividers (`#E2E8F0` / `#CBD5E1`)**: Subtle borders separating dense data streams without introducing heavy visual noise.
- **Text Layers**: 
  - Primary Headlines & Critical Telemetry: Slate-900 (`#0F172A`)
  - Body Content & Axis Data: Slate-700 (`#334155`)
  - Secondary Identifiers & Unit Labels: Slate-500 (`#64748B`)
  - Subdued / Disabled Indicators: Slate-400 (`#94A3B8`)

### Accent & Execution Accents
- **Primary Accent (`#0284C7`)**: Vivid electric azure cyan used sparingly for critical interactive triggers, running pipeline highlights, tab actives, and focal scalar readouts.

### Functional Status System
- **Converged / Optimal**: Emerald (`#059669` foreground, `#ECFDF5` container, `#A7F3D0` stroke)
- **Running / Iterating / Warning**: Amber (`#D97706` foreground, `#FFFBEB` container, `#FDE68A` stroke)
- **Infeasible / Solver Failure**: Rose (`#E11D48` foreground, `#FFF1F2` container, `#FECDD3` stroke)
- **Idle / Queued**: Slate (`#64748B` foreground, `#F1F5F9` container, `#CBD5E1` stroke)

## Typography

The type system is bifurcated strictly by functional domain: **Inter** handles structural UI labels, navigation, and qualitative data, while **JetBrains Mono** governs all numerical, temporal, scalar, and operational machine readouts.

- **Numerics & Matrix Alignment**: All iterations, matrix sizes, pivot tallies, dual gap percentages, and execution runtimes use JetBrains Mono with tabular lining figures enabled. This guarantees horizontal column alignment across table rows and live data feeds.
- **Metric Micro-Labels**: Standard operational telemetry headers use `label-caps` (Inter, uppercase, tracking +0.04em, Slate-500) paired immediately with `mono-metric-xl` or `mono-metric-md` values.
- **Readability Rules**: Body typography is set compact (12px–14px) with tighter-than-default line heights to preserve screen density while preventing line collisions.

## Layout & Spacing

The dashboard employs a fluid, edge-to-edge layout engine tailored for multi-monitor computational desks and technical control rooms.

- **Grid Architecture**: 12-column modular grid with standard `gutter` (16px) scaling to `gutter-desktop` (20px) on wide displays (≥1440px). Metrics summary strips utilize strict 4-column or 6-column subdivisions.
- **Dense Spacing Model**: Spacing tokens are compact. Core component padding and parameter row gaps rely heavily on `space-xs` (4px), `space-sm` (8px), and `space-md` (12px), eliminating superfluous vertical whitespace to maximize visual data throughput.
- **Layout Zones**:
  - **Top Chrome (48px fixed height)**: Global solver session status, dark/light theme switch, cluster node selector, and workspace breadcrumbs.
  - **Fixed Telemetry Ribbon**: Top-level KPIs (Objective value, MIP gap, Dual bound, Elapsed runtime, Simplex iterations).
  - **Main Console**: Split viewport featuring convergence charts/log stream (left/center) and constraint parameter inspector (right).

## Elevation & Depth

Visual hierarchy is maintained through high-precision structural boundaries rather than deep spatial shadows.

- **Surface Tiers**:
  - **Level 0 (Canvas)**: Background slate `#F8FAFC`.
  - **Level 1 (Card/Panel Base)**: Crisp `#FFFFFF` surfaces defined by a mandatory 1px border of `#E2E8F0` and an ultra-subtle ambient drop shadow (`0 1px 2px 0 rgba(15, 23, 42, 0.05)`).
  - **Level 2 (Hover & Active Panels)**: Border shifts to `#CBD5E1` with elevated ambient spread (`0 4px 6px -1px rgba(15, 23, 42, 0.07), 0 2px 4px -2px rgba(15, 23, 42, 0.05)`).
  - **Level 3 (Overlays & Slide-out Drawers)**: 1px border of `#CBD5E1` accompanied by a controlled drop shadow (`0 10px 15px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.03)`).
- **Inner Recesses**: Log monitors and matrix inspectors use an inset shallow border and `#F1F5F9` background to visually nest machine-level input/output beneath the UI container plane.

## Shapes

The design system follows a compact "Soft" corner system (`roundedness: 1`). 

- **Containers & Cards**: 4px (`0.25rem`) standard border radius. Large operational layout cards max out at 8px (`0.5rem`).
- **Interactive Controls**: Buttons, input boxes, status chips, and table row selections strictly use 4px corner rounding.
- **Rationale**: Minimal corner curvature maximizes usable internal pixel area for high-density plots and tables, establishing a strict, instrument-grade aesthetic appropriate for mathematical tooling.

## Components

### Buttons & Action Bars
- **Primary Button**: Solid Azure Cyan `#0284C7`, text `#FFFFFF`, 4px radius, font weight 600. Active states drop to `#0369A1`. Hover states use subtle glow: `0 0 0 1px #0284C7, 0 1px 2px rgba(2, 132, 199, 0.2)`.
- **Secondary / Solver Action Button**: Background `#FFFFFF`, border 1px solid `#CBD5E1`, text `#0F172A`. Hover transitions background to `#F8FAFC` and border to `#94A3B8`.
- **Icon Buttons (e.g., Dark Theme Toggle)**: Square 28px × 28px or 32px × 32px targets, borderless or 1px subtle stroke `#E2E8F0`, foreground `#64748B`, transitioning to `#0F172A` on hover.

### Status Chips & Pills
- **Geometry**: Compact 20px height, 4px border radius, 6px horizontal padding.
- **Variants**:
  - *Optimal*: Background `#ECFDF5`, border `#A7F3D0`, text `#065F46` (with a 6px solid `#059669` status dot).
  - *Iterating*: Background `#FFFBEB`, border `#FDE68A`, text `#92400E` (with a pulsating `#D97706` indicator).
  - *Infeasible*: Background `#FFF1F2`, border `#FECDD3`, text `#9F1239` (with a solid `#E11D48` dot).
  - *Idle*: Background `#F1F5F9`, border `#CBD5E1`, text `#475569`.

### Data Cards & Telemetry Blocks
- Pure white `#FFFFFF` surface enclosed by a 1px `#E2E8F0` border.
- Header zone: Uppercase micro-label (`label-caps`) in `#64748B` with optional inline unit indicator (e.g., `(ms)` or `(%)`).
- Main readout: `mono-metric-xl` in `#0F172A`. Sub-metric or change delta displayed directly beneath in `mono-data-sm` with semantic delta colors.

### Form Inputs & Solver Parameter Controls
- Height: Compact 30px standard.
- Background: `#FFFFFF`, border: 1px solid `#CBD5E1`, text: `JetBrains Mono` 12px for numerical fields, `Inter` 12px for textual variables. Focus state draws a crisp, unblurred 2px focus ring of `#0284C7` with zero offset.

### Tables & Log Streams
- **Data Tables**: Zero vertical borders; horizontal borders are 1px `#F1F5F9`. Header row sticky, background `#F8FAFC`, uppercase 11px font. Alternating row fills are avoided; hover states trigger a light `#F8FAFC` background.
- **Log Stream Component**: Background `#F8FAFC` with an inset 1px border `#E2E8F0`. Text rendered in `JetBrains Mono` 11px / 16px line height. Timestamp in `#94A3B8`, severity prefix colored per status tokens, message body in `#334155`.