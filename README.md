# ResponseCompleteWannier.jl
## すべてのJuliaをシングルテキストファイルにコピペするps1コマンド

`git ls-files "*.jl" | Sort-Object | ForEach-Object { "`n`n# ===== FILE: $_ =====`n"; Get-Content $_ -Raw } | Set-Content -Encoding utf8 ResponseCompleteWannier_all_julia.txt`
