# Implementation foundation and upstream API audit

## Architecture and scope

Use [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl) for plane-wave
ground states, Hamiltonians, Bloch orbitals, and suitable linear-response
operations. Use [Wannier.jl](https://github.com/qiaojunfeng/Wannier.jl) for
Wannierization, gauge matrices, overlaps, and Wannier interpolation.

The intended path is DFTK plane-wave reference calculations to a Wannier
subspace and response residuals, followed by response-complement compression
and, separately, dynamic Schur reduction and auxiliary linearization. The
project's contributions are response-complete spaces, PW Sternheimer teacher
calculations, response residuals and complement compression, dynamic Schur
extensions, structure-preserving model reduction and auxiliary linearization,
recovery of frequency-shift structure, multi-shift or frequency-recycling
Krylov methods, and Berry-connection improvements for response.

Do not implement another plane-wave DFT code, Wannierization algorithm, or
generic interpolation framework. Before adding a low-level routine, inspect
the installed upstream versions and their relevant source, identify existing
objects and operations, and prefer their native structures and thin adapters
to copied code. Implement locally only what the upstream libraries do not
provide suitably. Keep upstream-dependent code behind small adapters where
practical. Project-specific `PWReference`, `WannierSubspace`,
`ResponseComplement`, and `SchurAuxiliaryModel` concepts should remain
distinct where needed to express their mathematics; they must not duplicate
upstream data merely to wrap it.

## Source audit: released upstream versions

The following is a source inspection, **not** an installed-package or
numerical integration test. It concerns DFTK **v0.8.1**
([Project.toml](https://github.com/JuliaMolSim/DFTK.jl/blob/v0.8.1/Project.toml))
and Wannier **v0.3.6**
([Project.toml](https://github.com/qiaojunfeng/Wannier.jl/blob/v0.3.6/Project.toml)).
DFTK v0.8.1 declares Julia 1.10 compatibility and Wannier 0.3.2 compatibility,
which includes Wannier v0.3.6. Reinspect the actual installed versions before
implementing an adapter; these source references do not establish which
versions are installed or validated.

- DFTK provides `self_consistent_field` and its Wannier extension provides
  `Wannier.Model(scfres; ...)`. The latter writes Wannier90-format input files
  and reads them into a native Wannier model; it is not an in-memory adapter.
  See the [extension](https://github.com/JuliaMolSim/DFTK.jl/blob/v0.8.1/ext/DFTKWannierExt.jl)
  and its [integration test](https://github.com/JuliaMolSim/DFTK.jl/blob/v0.8.1/test/external/wannier.jl).
  Preserve the caller's control over filenames, projections and other options.
- Wannier's [`Model`](https://github.com/qiaojunfeng/Wannier.jl/blob/v0.3.6/src/model.jl)
  stores `U` with dimensions `(n_bands, n_wann, n_kpts)`, overlaps `M`, and
  k-point and band data. Its `disentangle` and `max_localize` routines operate
  on this model; Wannier also provides an
  [`InterpModel`](https://github.com/qiaojunfeng/Wannier.jl/blob/v0.3.6/src/interp/interp_model.jl).
  A gauge matrix is not itself a plane-wave basis: building one requires
  consistent Bloch states, band selection, k-point ordering and gauge.
- DFTK exports `apply_χ0(scfres, δV; ...)`, which uses its Sternheimer solver
  and returns density and orbital-response information for a nonzero
  perturbation. See [`chi0.jl`](https://github.com/JuliaMolSim/DFTK.jl/blob/v0.8.1/src/response/chi0.jl).
  The potential is normalized internally; `δρ` is rescaled but returned
  `δψ` is not. Its zero-potential path has a different return shape.
  Any teacher adapter must explicitly handle normalization, convergence,
  spin, k/q mapping and plane-wave basis compatibility. Do not treat this
  static interface as a validated dynamic-frequency Sternheimer solver.

No DFTK/Wannier adapter, plane-wave teacher, frequency convention, benchmark,
or physical validation is currently implemented in this repository. The
existing `ResponseComplement` and `SchurAuxiliaryModel` implement only the
synthetic matrix-level operations described in
[MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md).

## Reproducibility requirements for integration and benchmarks

For each integration step, inspect the installed DFTK and Wannier versions
and their source rather than guessing their APIs. Declare direct dependencies
and compatible versions in `Project.toml`, and record exact resolved versions
in the applicable `Manifest.toml`. A benchmark environment should retain its
own manifest when appropriate. Benchmark metadata must also record the exact
Julia, DFTK and Wannier versions and the project revision, alongside the
physical inputs, numerical parameters, perturbation and frequency
conventions, convergence status, and tolerances necessary to reproduce the
calculation. Do not report synthetic examples as physical benchmarks.

No upstream package versions are currently installed or pinned by this
project, and no manifest or benchmark metadata has yet been generated.
