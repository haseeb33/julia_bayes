global MINIMUM_PARAM = 10E-200

mutable struct Dirichlet
    K::Int64
    alpha::Array{Float64}
    sumAlpha::Float64
end

Dirichlet(param::Array) = Dirichlet(size(param)[1], param, sum(param))
Dirichlet(K::Int, a::Float64) = Dirichlet(K, [a for i in 1:K], K*a)

function dirichlet_optimizeParam(Ck, ndkMax, C_, ndMax, numIteration, dirichlet_obj::Dirichlet)
    function digammaRecurrence(nMax, C, z)
        if z==0.0
            return 0.0
        end
        
        R=0; S=0;
        for n in 1:nMax
            R+= 1.0 / (n-1+z)
            S+= C[n]*R
        end
        return S
    end
            
    for i in 1:numIteration
        demon = digammaRecurrence(ndMax, C_, dirichlet_obj.sumAlpha)
        
        for k in 1:dirichlet_obj.K
            numer = digammaRecurrence(ndkMax[k], Ck[k], dirichlet_obj.alpha[k])
            dirichlet_obj.alpha[k] *= (numer/demon)
            dirichlet_obj.alpha[k] = max(dirichlet_obj.alpha[k], MINIMUM_PARAM)
        end
        dirichlet_obj.sumAlpha = sum(dirichlet_obj.alpha)
    end
end

function dirichlet_set(param, dirichlet_obj::Dirichlet)
    for k in 1:dirichlet_obj.K
        dirichlet_obj.alpha[k] = param[k] > MINIMUM_PARAM ? param[k] : MINIMUM_PARAM
    end
end
