using Distributed
#addprocs(Sys.CPU_THREADS - 2) 

@everywhere begin 
    using Base.Iterators
    using Random
    using ABCDGraphGenerator
    using StatsBase
    using LinearAlgebra
    using Logging
    using PyCall
    using Clustering
    
    #auxilliary function - HOPE embedding:
    function Hope(g, sim, dim, fn; beta=0.01, alpha=0.5)
        dim = dim*2
        A = g.get_adjacency().data
        n = g.vcount()
        ## Katz
        if sim == :katz
            M_g = I - beta * A
            M_l = beta * A
        end
        ## Adamic-Adar
        if sim == :aa
            M_g = I
            D = diagm((x -> x > 1 ? 1/log(x) : 0.0).(g.degree()))
            M_l = A*D*A
            M_l[diagind(M_l)] .= 0.0
        end
        ## Common neighbors
        if sim == :cn
            M_g = I
            M_l = A*A
        end
        ## personalized page rank
        if sim == :ppr
            P = mapslices(A, dims=1) do x
                s = sum(x)
                iszero(s) ? fill(1/n, n) : x / s
            end
            M_g = I-alpha*P
            M_l = (1-alpha)*I
        end
        S = M_g \ M_l
        k = div(dim, 2)
        u, s, vt = svd(S)
        X1 = u[:, 1:k] * diagm(sqrt.(s[1:k]))
        ## undirected graphs have identical source and target embeddings
        if !g.is_directed()
            X = X1
        else
            X2 = vt[:, 1:k] * diagm(sqrt.(s[1:k]))
            X = [X1 X2]
        end


        open(fn, "w") do f
            println(f, size(X,1), " ", size(X, 2))
            for i in axes(X, 1)
                print(f, i, " ")
                for j in axes(X, 2)
                    print(f, X[i, j])
                     j < size(X, 2) && print(f, " ")
                end
                println(f)
            end
        end

    end
    

    #General Parameters:
    no_iters = 50
    seed = 1442
    Random.seed!(seed)
    
    #Paths:
    coms_dir = "experiment/coms/"
    degs_dir = "experiment/degs/"
    graphs_dir = "experiment/graphs/"
    embeds_dir = "experiment/embeddings/"
    res_dir = "experiment/results/"
    logs_dir = "experiment/logs/"
    julia_path = "julia-1.7.2/bin/julia"

    mkpath("experiment")
    mkpath(coms_dir)
    mkpath(degs_dir)
    mkpath(graphs_dir)
    mkpath(embeds_dir)
    mkpath(res_dir)
    mkpath(logs_dir);
        
    #Graph Parameters:
    ns = 10^3
    ξs = [0.15, 0.25, 0.5, 0.65, 0.75, 0.85]
    βs = [1.1, 1.5, 1.9]
    γs = [2.1, 2.5, 2.9]
    isCL = false        
    min_deg = [1,2,5]
    is_local = false
    
    graph_params = sort(reshape(collect(product(ns,ξs,βs,γs,isCL,min_deg,is_local)),:));
    
    #Embeddings parameters:
    dims = 2 .^ collect(3:Int(ceil(log(2,Int(ceil(0.2 *ns))))))


    #deepwalk/node2vec parameters:
    no_walks = 10
    walk_length = 100
    window_size = 10
    ps = [0.25, 0.50, 1., 2., 4.]
    qs = [0.25, 0.50, 1., 2., 4.];


    #LINE parameters:
    negative_ratios = 5

    #SDNE parameters:
    sdne_betas = collect(2:1:10);

    #GraRep parameters:
    grarep_k = [1,2,4,8]

    #HOPE parameters:
    sim_meausures = [:katz, :ppr, :cn, :aa]
    
end

@distributed for (n, ξ, β, γ, isCL, min_deg, is_local) in graph_params
    @info "Worker $(myid()) start working on parameters $(n) $(ξ) $(γ) $(β) $(min_deg)"
    
    #set params:
    c_min = Int(ceil(0.005*n))               
    c_max = Int(ceil(0.2*n)) 
    max_deg = Int(ceil(sqrt(n)))
    
    #import Python packages: Igraph and Leiden algorithm:
    nx = pyimport("networkx");
    community_louvain = pyimport("community"); 
    ig = pyimport("igraph");
    la = pyimport("leidenalg");
    ecg = pyimport("partition_igraph")

    #check if graph already exist:
    g_name = graphs_dir * "g_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg).dat"
    g_cl_name = graphs_dir * "g_clusters_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg).dat"
    logname = logs_dir * "g_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg).dat"
    if !isfile(g_name)
        #sample degree sequence:
        deg_filename = degs_dir * "deg_$(n)_$(min_deg)_$(max_deg)_$(γ).dat"
        if !isfile(deg_filename)
            degs = ABCDGraphGenerator.sample_degrees(γ, min_deg, max_deg, n, 1)
            open(io -> foreach(d -> println(io, d), degs), deg_filename, "w")
        else
                degs = parse.(Int,readlines(deg_filename))
        end
    
        #sample communities:
        com_filename = coms_dir * "com_$(n)_$(c_min)_$(c_max)_$(β).dat"
        if !isfile(com_filename)
            coms = ABCDGraphGenerator.sample_communities(β, c_min, c_max, n, 1)
            open(io -> foreach(d -> println(io, d), coms), com_filename, "w")
        else
            coms = parse.(Int,readlines(com_filename))
        end
        
        #generate graph:
        try
            p = ABCDGraphGenerator.ABCDParams(degs, coms, nothing, ξ, isCL, is_local)
            edges, clusters = ABCDGraphGenerator.gen_graph(p)
            open(g_name, "w") do io
                for (a, b) in sort!(collect(edges))
                    println(io, a, "\t", b)
                end
            end
            open(g_cl_name, "w") do io
                for (i, c) in enumerate(clusters)
                    println(io, i, "\t", c)
                end   
            end
       catch err 
            logger = SimpleLogger(open(logname, "w+"))
            with_logger(logger) do
                @warn "graph generation error:"
                @warn exception = (err, stacktrace()), 
                "for params: $(n) $(ξ) $(γ) $(β) $(min_deg)"
            end
        end
    end
    #calculate baseline stats:
    try 
        #read graph (if exist):
        G = ig.Graph.Read_Edgelist(g_name, directed = false)
        G.delete_vertices(0)
        g = nx.read_edgelist(g_name)
        nodeslist = collect(nx.nodes(g));
        
        partition = [parse.(Int,e2) - 1 for (e1,e2) in split.(readlines(g_cl_name))]
        no_partitions = length(unique(partition))
        ground_truth_modularity = G.modularity(partition)
        
        measures = Dict("louvain_modularity" => [],
                "louvain_ami" => [],
                "leiden_modularity" => [],
                "leiden_ami" => [],
                "ecg_modularity" => [],
                "ecg_ami" => [],)
        #louvain_lengths = []
        #leiden_lengths = []
        #ecg_lengths = []
        #find partitions:
        for i = 1:no_iters
            #louvain stats:
            louvain_partition = community_louvain.best_partition(g, random_state = i)
            #louvain_length = length(unique(values(louvain_partition)))
            #push!(louvain_lengths, louvain_length)
            louvain_modularity = community_louvain.modularity(louvain_partition,g)   
            push!(measures["louvain_modularity"], louvain_modularity)
            louvain_partition = [louvain_partition["$j"] for j = 1:length(louvain_partition)]
            ami_gt_louvain = mutualinfo(partition,louvain_partition)
            push!(measures["louvain_ami"], ami_gt_louvain)
            
            #leiden stats:
            leiden_partition = la.find_partition(G, la.ModularityVertexPartition, seed = i)
            #leiden_length = length(unique(leiden_partition.membership))
            #push!(leiden_lengths, leiden_length)
            leiden_modularity = G.modularity(leiden_partition.membership)
            push!(measures["leiden_modularity"], leiden_modularity)
            ami_gt_leiden = mutualinfo(partition,leiden_partition.membership)
            push!(measures["leiden_ami"], ami_gt_leiden)
            
            #ECG stats:
            ecg_partition = G.community_ecg(ens_size=32)
            #ecg_length = length(unique(ecg_partition.membership))
            #push!(ecg_lengths, ecg_length)
            ecg_modularity = G.modularity(ecg_partition.membership)
            push!(measures["ecg_modularity"], ecg_modularity)
            ami_gt_ecg = mutualinfo(partition,ecg_partition.membership)
            push!(measures["ecg_ami"], ami_gt_ecg)
        end
           
        #save to the files:
        for algo in ["louvain", "leiden", "ecg"]
            for measure in ["modularity", "ami"]
                stats_name = res_dir * "g_stats_$(algo)_$(measure)_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg).dat"
                open(stats_name, "w") do io
                    println(io, join(measures["$(algo)_$(measure)"],";"))
                end      
            end
        end    
    catch err
        logger = SimpleLogger(open(logname, "w+"))
        with_logger(logger) do
            @warn "cannot calculate stats:"
            @warn exception = (err, stacktrace()), 
            "for params: $(n) $(ξ) $(γ) $(β) $(min_deg)"
        end
    end
    #time for embeddings: 
    # Locally Linear Embedding:
    for dim in dims
        try
            output_name = embeds_dir * "LLE_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg)_$(dim).dat"
            cmd = `python -m openne --method lle --input $(g_name) --graph-format edgelist 
            --output $(output_name) --representation-size $(dim)`
            run(cmd)
            cmd = `$(julia_path) run_clustering.jl $(output_name) $(g_name) $(g_cl_name) $(res_dir) $(seed)`
            run(cmd)
        catch err
            logger = SimpleLogger(open(logname, "w+"))
            with_logger(logger) do
                @warn "LLE embedding error:"
                @warn exception = (err, stacktrace()), 
                "for params: $(n) $(ξ) $(γ) $(β) $(min_deg)"
            end
        end
    end
    
    # Laplacian Eigenmaps:
    for dim in dims
        try
            output_name = embeds_dir * "LE_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg)_$(dim).dat"
            cmd = `python -m openne --method lap --input $(g_name) --graph-format edgelist 
            --output $(output_name) --representation-size $(dim)`
            run(cmd)
            cmd = `$(julia_path) run_clustering.jl $(output_name) $(g_name) $(g_cl_name) $(res_dir) $(seed)`
            run(cmd)
        catch err
            logger = SimpleLogger(open(logname, "w+"))
            with_logger(logger) do
                @warn "LE embedding error:"
                @warn exception = (err, stacktrace()), 
                "for params: $(n) $(ξ) $(γ) $(β) $(min_deg)"
            end
        end
    end

    #deepWalk:
    for (dim, walks, ℓ, window) in sort(reshape(collect(product(dims,no_walks,walk_length,window_size)),:))
      try
            output_name = embeds_dir * "deepWalk_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg)_$(dim).dat"
            cmd = `python -m openne --method deepWalk --input $(g_name) --graph-format edgelist 
            --output $(output_name) --representation-size $(dim) --number-walks $(walks) --walk-length $(ℓ)
            --window-size $(window)`
            run(cmd)
            cmd = `$(julia_path) run_clustering.jl $(output_name) $(g_name) $(g_cl_name) $(res_dir) $(seed)`
            run(cmd)
        catch err
            logger = SimpleLogger(open(logname, "w+"))
            with_logger(logger) do
                @warn "DeepWalk embedding error:"
                @warn exception = (err, stacktrace()), 
                "for params: $(n) $(ξ) $(γ) $(β) $(min_deg)"
            end
        end
    end
    
    #node2vec:
    for (dim, walks, ℓ, window, p, q) in sort(reshape(collect(product(dims,no_walks,walk_length,window_size,ps,qs)),:));
      try
            output_name = embeds_dir * "node2vec_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg)_$(dim)_$(p)_$(q).dat"
            cmd = `python -m openne --method node2vec --input $(g_name) --graph-format edgelist 
            --output $(output_name) --representation-size $(dim) --number-walks $(walks) --walk-length $(ℓ)
            --window-size $(window) --q $(q) --p $(p)`
            run(cmd)
            cmd = `$(julia_path) run_clustering.jl $(output_name) $(g_name) $(g_cl_name) $(res_dir) $(seed)`
            run(cmd)
        catch err
            logger = SimpleLogger(open(logname, "w+"))
            with_logger(logger) do
                @warn "node2vec embedding error:"
                @warn exception = (err, stacktrace()), 
                "for params: $(n) $(ξ) $(γ) $(β) $(min_deg)"
            end
        end
    end

    #LINE:
    for (dim, negative_ratio) in sort(reshape(collect(product(dims,negative_ratios)),:));
      try
            output_name = embeds_dir * "LINE_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg)_$(dim).dat"
            cmd = `python -m openne --method line --input $(g_name) --graph-format edgelist 
            --output $(output_name) --representation-size $(dim) --negative-ratio $(negative_ratio)`
            run(cmd)
            cmd = `$(julia_path) run_clustering.jl $(output_name) $(g_name) $(g_cl_name) $(res_dir) $(seed)`
            run(cmd)
        catch err
            logger = SimpleLogger(open(logname, "w+"))
            with_logger(logger) do
                @warn "LINE embedding error:"
                @warn exception = (err, stacktrace()), 
                "for params: $(n) $(ξ) $(γ) $(β) $(min_deg)"
            end
        end
    end

    #SDNE:
    for (dim, beta) in sort(reshape(collect(product(dims,sdne_betas)),:));
        for ℓ in 2 .^ collect(Int(ceil(log(2,Int(ceil(0.125 *ns))))):Int(ceil(log(2,Int(ceil(0.5 *ns))))))
            ℓ ≤ dim && continue 
            encoder_list = "[$ℓ,$dim]"
            try
                output_name = embeds_dir * "SDNE_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg)_$(dim)_$(ℓ)_$(beta).dat"
                cmd = `python -m openne --method sdne --input $(g_name) --graph-format edgelist 
                --output $(output_name) --representation-size $(dim) --encoder-list $(encoder_list) --beta $(beta)`
                run(cmd)
                cmd = `$(julia_path) run_clustering.jl $(output_name) $(g_name) $(g_cl_name) $(res_dir) $(seed)`
                run(cmd)
            catch err
                logger = SimpleLogger(open(logname, "w+"))
                with_logger(logger) do
                    @warn "SDNE embedding error:"
                    @warn exception = (err, stacktrace()), 
                    "for params: $(n) $(ξ) $(γ) $(β) $(min_deg)"
                end
            end
        end
    end

    #GraRep:
    for (dim,k) in sort(reshape(collect(product(dims,grarep_k)),:));
        try
            output_name = embeds_dir * "GraRep_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg)_$(dim)_$(k).dat"
            cmd = `python -m openne --method grarep --input $(g_name) --graph-format edgelist 
                --output $(output_name) --representation-size $(dim) --kstep $(k)`
            run(cmd)
            cmd = `$(julia_path) run_clustering.jl $(output_name) $(g_name) $(g_cl_name) $(res_dir) $(seed)`
            run(cmd)
        catch err
            logger = SimpleLogger(open(logname, "w+"))
            with_logger(logger) do
                @warn "GraRep embedding error:"
                @warn exception = (err, stacktrace()), 
                "for params: $(n) $(ξ) $(γ) $(β) $(min_deg)"
            end
        end
        
    end

    #HOPE:
    for (dim,sim) in sort(reshape(collect(product(dims,sim_meausures)),:));
        try
            G = ig.Graph.Read_Edgelist(g_name, directed = false)
            G.delete_vertices(0)
            output_name = embeds_dir * "HOPE_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg)_$(dim)_$(sim).dat"
            Hope(G, sim, dim, output_name)
            cmd = `$(julia_path) run_clustering.jl $(output_name) $(g_name) $(g_cl_name) $(res_dir) $(seed)`
            run(cmd)
        catch err
            logger = SimpleLogger(open(logname, "w+"))
            with_logger(logger) do
                @warn "HOPE embedding error:"
                @warn exception = (err, stacktrace()), 
                "for params: $(n) $(ξ) $(γ) $(β) $(min_deg)"
            end
        end
        
    end

    @info "Worker $(myid()) finished working on parameters $(n) $(ξ) $(γ) $(β) $(min_deg)"
end
