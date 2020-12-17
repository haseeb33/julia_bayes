function Sampler_sample(p)
    u = rand()*sum(p)
    for i in 1:size(p)[1]
        if u<=p[i]
            return i
        end
        u-=p[i]
    end
    return size(p)[1]
end