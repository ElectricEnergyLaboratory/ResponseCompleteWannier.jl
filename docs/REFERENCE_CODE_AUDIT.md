# Reference-code audit: `kemlin_spectrum.jl`

## Status and scope

The requested source,
[`KemlinExtended/kemlin_spectrum.jl`](https://huggingface.co/datasets/takayukimakino/OpenRepos/blob/main/KemlinExtended/kemlin_spectrum.jl),
is not present in this checkout. Attempts to retrieve its Hugging Face file
page, direct file URL, and dataset API failed because `huggingface.co` could
not be resolved from this environment. GitHub code search did not locate a
copy. The local repository contains only its title-only README and therefore
does not provide an alternate implementation to inspect.

Consequently, this document is an **audit status record, not a completed
implementation audit**. No equation, convention, function location, shape, or
behavior below is inferred from the filename or problem description. The
prototype must be obtained and inspected before an adapter is written or its
physical conventions are changed.

## Repository baseline

At inspection time, the checkout had only `README.md`; it had no Julia source,
package metadata, tests, or documentation. Thus the current repository itself
does not implement DFTK ground-state setup, Sternheimer response, Schur
elimination, dielectric response, q/G indexing, or an optical limit.

## Requested source audit inventory

| Item | Audit result |
|---|---|
| Ground-state DFTK setup | Not audited: source unavailable. |
| Occupied states | Not audited: source unavailable. |
| Extra states | Not audited: source unavailable. |
| Projectors | Not audited: source unavailable. |
| Static Sternheimer implementation | Not audited: source unavailable. |
| Dynamic Sternheimer implementation | Not audited: source unavailable. |
| Frequency-shift convention | Not audited: source unavailable; do not assume the sign of `ω` or `iη`. |
| Eta convention | Not audited: source unavailable; do not assume whether `η` is added or subtracted. |
| Schur-complement implementation | Not audited: source unavailable; no code location can be named. |
| Density-response construction | Not audited: source unavailable. |
| q-dependent response | Not audited: source unavailable. |
| χ₀ construction | Not audited: source unavailable. |
| Dielectric matrix | Not audited: source unavailable. |
| Macroscopic dielectric function | Not audited: source unavailable. |
| q → 0 treatment | Not audited: source unavailable; no expansion or optical convention is presumed. |
| Current spectrum output | Not audited: source unavailable. |
| Gauge assumptions | Not audited: source unavailable. |
| Repeated frequency-by-frequency work | Not audited: source unavailable. |

## Questions to resolve when the source is available

The completed audit must record file/line locations, equations, and actual
array dimensions/indexing for each requested item. In particular:

1. Trace the orbital data produced by the PW response and determine whether
   the wavefunctions remain available in a form that can be projected into a
   Wannier subspace.
2. Record k, q, reciprocal-vector, band, and frequency axis ordering rather
   than assuming a conventional Julia layout.
3. Identify the precise density and χ₀ contractions, dielectric-matrix
   inversion, macroscopic averaging, and treatment of the `q → 0` limit.
4. Determine whether `U(k)`, neighbor overlaps/MMN, `H(R)`, Wannier centers,
   Berry connections, or velocity matrix elements are already computed or
   serialized.
5. Identify missing metadata needed to construct and validate PW response
   residuals, including compatible basis representation and the specific
   perturbation/frequency labels for each response column.
6. Preserve an independently callable PW reference path and document its
   conventions before connecting any reduced-space implementation.

Until that review is possible, the new Julia code is limited to synthetic
linear-algebra tests. It does not adapt, call, or modify the reference
prototype.
