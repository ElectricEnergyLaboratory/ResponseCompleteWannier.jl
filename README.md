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

このパッケージは、応答完全（response-complete）なワニエ空間と動的シューアモデルに関する研究プロトタイプです。
DFTK.jlを平面波DFTおよび線形応答の基盤とし、Wannier.jlをワニエ化および補間の基盤とすることを想定しています。
統合ポリシー、検証済みの上流エントリーポイント、および現時点での制限事項については、実装基盤（implementation foundation）と上流API監査を参照してください。

既存のコードは、合成的な密行列による残差圧縮とシューア代数を実装したものであり、まだいずれの上流パッケージにも接続されておらず、平面波計算に対する検証も行われていません。

## すべてのJuliaをシングルテキストファイルにコピペするps1コマンド

```
git ls-files "*.jl" | Sort-Object | ForEach-Object { "`n`n# ===== FILE: $_ =====`n"; Get-Content $_ -Raw } | Set-Content -Encoding utf8 ResponseCompleteWannier_all_julia.txt
```
