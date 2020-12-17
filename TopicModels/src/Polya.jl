mutable struct Polya
    K::Int64
    dir::Dirichlet
    n::Array{Int64}
    N::Int64
end

Polya(param::Dirichlet) = Polya(param.K, param, [0 for i in 1:param.K], 0)

function polya_p(x::Int, polya_obj::Polya)
    return (polya_obj.n[x]+polya_obj.dir.alpha[x]) / (polya_obj.N+polya_obj.dir.sumAlpha)
end

function polya_p(X::Array, polya_obj::Polya)
    p = 1.0
    for x in X
        p*= polya_p(x, polya_obj)
        polya_observe(x, polya_obj)
    end
    for x in X
        polya_forget(x, polya_obj)
    end
    return p
end
  
function polya_observe(x::Int, polya_obj::Polya)
    polya_obj.n[x]+=1
    polya_obj.N+=1  
end

function polya_observe(X::Array, polya_obj::Polya)
    for x in X
        polya_observe(x, polya_obj)
    end
end

function polya_forget(x::Int, polya_obj::Polya)
    polya_obj.n[x] -= 1
    polya_obj.N -= 1
end

function polya_forget(X::Array, polya_obj::Polya)
    for x in X
        polya_forget(x, polya_obj)
    end
end

function polya_getCount(k::Int, polya_obj::Polya)
    return polya_obj.n[k]
end