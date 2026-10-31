# Fix for capture branch in _select_naux
# Original convoluted version simplified to:

if rank_bound == 0 || iszero(last(cumulative))
    retained = 0
else
    retained = something(findfirst(>=(capture - slack), cumulative), rank_bound)
end

# The earlier version had duplicate logic:
# first_index = iszero(rank_bound) || iszero(last(cumulative)) ? nothing :
#               findfirst(>=(capture - slack), cumulative)
# retained = isnothing(first_index) ? (iszero(last(cumulative, 1)[1]) ? 0 : rank_bound) :
#            first_index
# if rank_bound == 0 || iszero(last(cumulative))
#     retained = 0
# else
#     retained = something(findfirst(>=(capture - slack), cumulative), rank_bound)
# end
# Simplified to the second if/else only.
