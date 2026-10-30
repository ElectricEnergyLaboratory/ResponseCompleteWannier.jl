# ResponseCompleteWannier.jl

This package is a research prototype for response-complete Wannier spaces and
dynamic Schur models. DFTK.jl is the intended plane-wave DFT and
linear-response foundation, and Wannier.jl is the intended Wannierization and
interpolation foundation. See [the implementation foundation and upstream
API audit](docs/IMPLEMENTATION_FOUNDATION.md) for the integration policy,
verified upstream entry points, and current limitations.

The existing code implements synthetic dense-matrix residual compression and
Schur algebra; it is not yet connected to either upstream package and has not
been validated against a plane-wave calculation.

## すべてのJuliaをシングルテキストファイルにコピペするps1コマンド

`git ls-files "*.jl" | Sort-Object | ForEach-Object { "`n`n# ===== FILE: $_ =====`n"; Get-Content $_ -Raw } | Set-Content -Encoding utf8 ResponseCompleteWannier_all_julia.txt`
