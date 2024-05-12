#run_clustering.jl embed_file g_name g_cl_name res_dir seed 

julia_path = "julia-1.7.2/bin/julia"
no_iters = 50
embed_file = ARGS[1]
params = split(embed_file[1:end-4], "_")
embed_name = split(params[1],"/")[end]

#check if graph is ABCD:

if tryparse(Int,params[2]) != nothing
    n = parse(Int,params[2])
    ξ = parse(Float64,params[3])
    γ = parse(Float64,params[4])
    β = parse(Float64,params[5])
    min_deg = parse(Int,params[6])
    dims = parse(Int,params[7])

    embed_params = ""
    if embed_name == "node2vec" 
        embed_params = "p: $(params[8]), q: $(params[9])"
    elseif embed_name == "SDNE"
        embed_params = "ℓ: $(params[8]), β: $(params[9])"
    elseif embed_name == "GraRep"
        embed_params = "k: $(params[8])"
    elseif embed_name == "HOPE"
        embed_params = params[8]
    end
else 
    graph_name = params[2]
    dims = parse(Int,params[3])

    embed_params = ""
    if embed_name == "node2vec" 
        embed_params = "p: $(params[4]), q: $(params[5])"
    elseif embed_name == "SDNE"
        embed_params = "ℓ: $(params[4]), β: $(params[5])"
    elseif embed_name == "GraRep"
        embed_params = "k: $(params[4])"
    elseif embed_name == "HOPE"
        embed_params = params[4]
    end
end
    
g_name = ARGS[2]
g_cl_name = ARGS[3]
res_dir = ARGS[4]
seed = parse(Int, ARGS[5])

using Random, Base.Iterators
using PyCall, Clustering
 
Random.seed!() = seed

nx = pyimport("networkx");
ig = pyimport("igraph");
community_louvain = pyimport("community"); 
la = pyimport("leidenalg");
gmm = pyimport("sklearn.mixture")
hdbscan = pyimport("hdbscan")

#embedding parameters:

#create results path:
mkpath(res_dir)

#read graph:
G = ig.Graph.Read_Edgelist(g_name, directed = false)
G.delete_vertices(0)
g = nx.read_edgelist(g_name)
nodeslist = collect(nx.nodes(g));
n = length(nodeslist)

g_cl_name != "nothing" && gt_partition = [parse.(Int,e2) - 1 for (e1,e2) in split.(readlines(g_cl_name))]

#Read embedding:
embed = transpose(reduce(hcat,[parse.(Float64,x) for x in split.(readlines(embed_file)[2:end])]))
embed = embed[sortperm(embed[:, 1]), :][:,2:end];

#CGE score:
CGE_score = readchomp(`$julia_path CGE_CLI.jl -g $(g_name) -e $(embed_file) -c $(g_cl_name) --seed $(seed)`)
CGE_score = [parse(Float64, x)  for x in split.(CGE_score[2:end-1], ", ")]

#K-Means and Gaussian Mixture Model:
n_clusters = [Int(ceil(n/k))   for k in 2 .^ collect(6:1:12) if k ≤ 0.2 * n]

#HDBSCAN:
min_samples = 1:10
    
    
#Mini Batch K-Means:
for k in n_clusters
    if g_cl_name != "nothing"
        measures = Dict("louvain_modularity" => [],
                        "louvain_ami" => [],
                        "leiden_modularity" => [],
                        "leiden_ami" => [])
    else
        measures = Dict("louvain_modularity" => [],
                    "leiden_modularity" => [])
    end
    labels = kmeans(transpose(embed),k).assignments
    d = Dict(nodeslist[i] => labels[i] for i = 1:length(labels))
    for i = 1:no_iters
        #louvain:
        louvain_partition = community_louvain.best_partition(g,d,random_state = i)
        louvain_modularity = community_louvain.modularity(louvain_partition,g)
        push!(measures["louvain_modularity"], louvain_modularity)
        if g_cl_name != "nothing"
            louvain_partition =  [louvain_partition["$j"] + 1 for j = 1:length(louvain_partition)]
            louvain_ami = mutualinfo(gt_partition, louvain_partition)
            push!(measures["louvain_ami"], louvain_ami)
        end
        
        #leiden:
        leiden_partition = la.find_partition(G, la.ModularityVertexPartition,initial_membership = labels, seed = i)
        leiden_modularity = G.modularity(leiden_partition.membership)
        push!(measures["leiden_modularity"], leiden_modularity)
        if g_cl_name != "nothing"
            leiden_ami = mutualinfo(gt_partition, leiden_partition.membership)
            push!(measures["leiden_ami"], leiden_ami)
        end
    end
    for algo in ["louvain", "leiden"]
        for measure in ["modularity", "ami"]
            (g_cl_name != "nothing" && measure == "ami") && continue
            if g_cl_name != "nothing"
                res_name = res_dir * "results_$(algo)_$(measure)_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg).dat"
            else
                res_name = res_dir * "results_$(algo)_$(measure)_$(graph).dat"
            end
            resline = vcat([embed_name, dims, embed_params, "K-Means", "k: $(k)"], 
                CGE_score, measures["$(algo)_$(measure)"])
            open(res_name, "a") do io
                println(io, join(resline,";"))
            end
        end
    end
end


#HDBSCAN
for ms in min_samples
    if g_cl_name != "nothing"
        measures = Dict("louvain_modularity" => [],
                        "louvain_ami" => [],
                        "leiden_modularity" => [],
                        "leiden_ami" => [])
    else
        measures = Dict("louvain_modularity" => [],
                    "leiden_modularity" => [])
    end
    c = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=ms).fit(embed)
    remain_clusters =  collect((length(unique(c.labels_)) - 1):length(c.labels_))
    labels = [c.labels_[i] != -1  ? c.labels_[i] : popfirst!(remain_clusters)
              for i = 1:length(c.labels_)]
    d = Dict(nodeslist[i] => labels[i] for i = 1:length(labels))
    for i = 1:no_iters
        #louvain:
        louvain_partition = community_louvain.best_partition(g,d,random_state = i)
        louvain_modularity = community_louvain.modularity(louvain_partition,g)
        push!(measures["louvain_modularity"], louvain_modularity)
        if g_cl_name != "nothing"
            louvain_partition =  [louvain_partition["$j"] + 1 for j = 1:length(louvain_partition)]
            louvain_ami = mutualinfo(gt_partition, louvain_partition)
            push!(measures["louvain_ami"], louvain_ami)
        end
        
        #leiden:
        leiden_partition = la.find_partition(G, la.ModularityVertexPartition,initial_membership = labels, seed = i)
        leiden_modularity = G.modularity(leiden_partition.membership)
        push!(measures["leiden_modularity"], leiden_modularity)
        if g_cl_name != "nothing"
            leiden_ami = mutualinfo(gt_partition, leiden_partition.membership)
            push!(measures["leiden_ami"], leiden_ami)
        end
    end
    for algo in ["louvain", "leiden"]
        for measure in ["modularity", "ami"]
            (g_cl_name != "nothing" && measure == "ami") && continue
            if g_cl_name != "nothing"
                res_name = res_dir * "results_$(algo)_$(measure)_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg).dat"
            else
                res_name = res_dir * "results_$(algo)_$(measure)_$(graph).dat"
            end
            resline = vcat([embed_name, dims, embed_params, "HDBSCAN", "min_samples: $(ms)"], 
                CGE_score, measures["$(algo)_$(measure)"])
            open(res_name, "a") do io
                println(io, join(resline,";"))
            end
        end
    end
end


#Gaussian Mixture Models
for k in n_clusters
   if g_cl_name != "nothing"
        measures = Dict("louvain_modularity" => [],
                        "louvain_ami" => [],
                        "leiden_modularity" => [],
                        "leiden_ami" => [])
    else
        measures = Dict("louvain_modularity" => [],
                    "leiden_modularity" => [])
    end
    c = gmm.GaussianMixture(n_components=k, random_state = seed).fit(embed)
    labels = c.predict(embed)
    d = Dict(nodeslist[i] => labels[i] for i = 1:length(labels))
    for i = 1:no_iters
        #louvain:
        louvain_partition = community_louvain.best_partition(g,d,random_state = i)
        louvain_modularity = community_louvain.modularity(louvain_partition,g)
        push!(measures["louvain_modularity"], louvain_modularity)
        louvain_partition =  [louvain_partition["$j"] + 1 for j = 1:length(louvain_partition)]
        louvain_ami = mutualinfo(gt_partition, louvain_partition)
        push!(measures["louvain_ami"], louvain_ami)
        
        #leiden:
        leiden_partition = la.find_partition(G, la.ModularityVertexPartition,initial_membership = labels, seed = i)
        leiden_modularity = G.modularity(leiden_partition.membership)
        push!(measures["leiden_modularity"], leiden_modularity)
        if g_cl_name != "nothing"
            leiden_ami = mutualinfo(gt_partition, leiden_partition.membership)
            push!(measures["leiden_ami"], leiden_ami)
        end
    end
    for algo in ["louvain", "leiden"]
        for measure in ["modularity", "ami"]
            (g_cl_name != "nothing" && measure == "ami") && continue 
            if g_cl_name != "nothing"
                res_name = res_dir * "results_$(algo)_$(measure)_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg).dat"
            else
                res_name = res_dir * "results_$(algo)_$(measure)_$(graph).dat"
            end
            resline = vcat([embed_name, dims, embed_params, "GMM", "k: $(k)"], 
                CGE_score, measures["$(algo)_$(measure)"])
            open(res_name, "a") do io
                println(io, join(resline,";"))
            end
        end
    end
end


