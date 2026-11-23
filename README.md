# Difference Returns — 3D

A self-contained interactive visualization of non-identical recurrence using the **Lorenz attractor**, a deterministic nonlinear dynamical system.

**Live website:** https://biohazardfish.github.io/difference-returns-3d/

The [interactive webpage](index.html) uses an HTML Canvas renderer with fourth-order Runge–Kutta (RK4) numerical integration. No build tools or JavaScript libraries are required.

## Controls

- Drag or swipe to orbit the 3D projection.
- Mouse wheel to zoom.
- Pause / Play, Reset, and Speed controls.
- Poincaré section plots upward crossings at z = 25.

## Python study

[`difference_returns_3d.py`](difference_returns_3d.py) preserves the Python animation generator; its output video is not required for the browser simulation.

## Philosophical distinction

The Lorenz system is **only an analogy**. Deleuze's reading of Nietzsche's eternal return concerns difference, becoming, affirmation, and selection; it is not equivalent to chaos theory, to a circular repetition, or to a trajectory that merely never closes.

## Deployment

The GitHub Actions workflow in `.github/workflows/pages.yml` publishes the static site using GitHub Pages.

MIT-style permission for reuse is not asserted; copyright remains with repository owner until a license is selected.
