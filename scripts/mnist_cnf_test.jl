using DrWatson
using InvertibleNetworks
using Random
using Flux
using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM
using LinearAlgebra
using PyPlot
using MLDatasets
using Statistics

# Function to visualize latent space in 2d instead
# Because network outputs a 1D vector
function z_shape_simple(G, ZX_test)
    Z_save, ZX = split_states(ZX_test[:], G.Z_dims)
    for i=G.L:-1:1
        if i < G.L
            ZX = tensor_cat(ZX, Z_save[i])
        end
        ZX = G.squeezer.inverse(ZX) 
    end
    ZX
end

# Random seed
Random.seed!(20)

# Plotting dir
exp_name = "train-mnist"
save_dict = @strdict exp_name
save_path = plotsdir(savename(save_dict; digits=6))

# Training hyperparameters
device = gpu
nepochs    = 300
batch_size = 128
lr        = 5f-4
lr_step   = 10
λ = 1f-1
noiseLev   = 0.01f0 # Additive noise


# Number of digist from mnist to train on 
num_digit = 1
range_dig = 0:num_digit-1

# load in training data
train_x, train_y = MNIST.traindata()

# grab the digist to train on 
inds = findall(x -> x in range_dig, train_y)
train_y = train_y[inds]
train_x = train_x[:,:,inds]

# Use a subset of total training data
num_samples = batch_size*45

# Number of training examples seen during training
total_seen = nepochs*num_samples

# Reshape data to 4D tensor with channel length = 1 in penultimate dimension
X_train = reshape(train_x, size(train_x)[1], size(train_x)[2], 1, size(train_x)[3])
X_train = Float32.(X_train[:,:,:,1:num_samples])
X_train = permutedims(X_train, [2, 1, 3, 4])

# load in test data
test_x, test_y = MNIST.testdata()
inds = findall(x -> x in range_dig, test_y)
test_y = test_y[inds]
test_x = test_x[:,:,inds]

X_test   = Float32.(reshape(test_x, size(test_x)[1], size(test_x)[2], 1, size(test_x)[3]))
X_test = permutedims(X_test, [2, 1, 3, 4])

nx, ny, nc, n_samples = size(X_train);
N = nx*ny 

# Create network
L = 2
K = 6
n_hidden = 64
low = 0.5f0

G = NetworkGlow(1, n_hidden, L, K; split_scales=true, activation=SigmoidLayer(low=low,high=1.0f0)) |> device

# Test batches 
X_train_latent = X_train[:,:,:,1:batch_size];
X_test_latent  = X_test[:,:,:,1:batch_size];

X_train_latent .+= noiseLev*randn(Float32, size(X_train_latent));
X_test_latent  .+= noiseLev*randn(Float32, size(X_test_latent));

X_train_latent = X_train_latent |> device;
X_test_latent  = X_test_latent  |> device;

# Noise for generative samples 
ZX_noise = randn(Float32, nx, ny, nc, batch_size) |> device;

# Split in training/testing
#use all as training set because there is a separate testing set
train_fraction = 1
ntrain = Int(floor((n_samples*train_fraction)))
nbatches = cld(ntrain, batch_size)

# Optimizer
#opt = Optimiser(ExpDecay(lr, .99f0, nbatches*lr_step, 1f-6), ADAM(lr))
opt = Optimiser(ADAM(lr))

# Training log keeper
floss = zeros(Float32, nbatches, nepochs);
flogdet = zeros(Float32, nbatches, nepochs);

floss_test = zeros(Float32, nepochs);
flogdet_test = zeros(Float32, nepochs);

# Same parameters every nth epoch
intermediate_save_params = nepochs

plot_every = 2

for e=1:nepochs
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)
    for b = 1:25#nbatches # batch loop
        X = X_train[:, :, :, idx_e[:,b]]
        X .+= noiseLev*randn(Float32, size(X))
        X = X |> device

        Zx, lgdet = G.forward(X)

        floss[b,e]   = norm(Zx)^2 / (N*batch_size)
        flogdet[b,e] = lgdet / (-N) # logdet is already normalized by batch_size

        G.backward((Zx / batch_size)[:], (Zx)[:])

        print("Iter: epoch=", e, "/", nepochs, ", batch=", b, "/", nbatches, 
            "; f l2 = ",  floss[b,e], 
            "; lgdet = ", flogdet[b,e], "; f = ", floss[b,e] + flogdet[b,e], "\n")
        for p in get_params(G)
            update!(opt, p.data, p.grad)
        end
        clear_grad!(G)

        Base.flush(Base.stdout)
    end

    if mod(e,plot_every) ==0
        ############# Test Image generation
        # Evaluate network on test dataset
        ZX_test, lgdet_test = G.forward(X_test_latent ) |> cpu;
        ZX_test_sq = z_shape_simple(G, ZX_test);

        flogdet_test[e] = lgdet_test / (-N)
        floss_test[e] = norm(ZX_test)^2f0 / (N*batch_size);

        # Evaluate network on train dataset
        ZX_train = G.forward(X_train_latent)[1] |> cpu;
        ZX_train_sq = z_shape_simple(G, ZX_train);

        #### make figures of generative samples
        X_gen = G.inverse(ZX_noise[:]) |> cpu;

        # Plot latent vars and qq plots. 
        mean_train_1 = round(mean(ZX_train_sq[:,:,1,1]),digits=2)
        std_train_1 = round(std(ZX_train_sq[:,:,1,1]),digits=2)

        mean_test_1 = round(mean(ZX_test_sq[:,:,1,1]),digits=2)
        std_test_1 = round(std(ZX_test_sq[:,:,1,1]),digits=2)

        mean_train_2 = round(mean(ZX_train_sq[:,:,1,2]),digits=2)
        std_train_2 = round(std(ZX_train_sq[:,:,1,2]),digits=2)

        mean_test_2 = round(mean(ZX_test_sq[:,:,1,2]),digits=2)
        std_test_2 = round(std(ZX_test_sq[:,:,1,2]),digits=2)

        fig = figure(figsize=(14, 12))
        subplot(4,5,1); imshow(X_gen[:,:,1,1], aspect=1, vmin=0,vmax=1,interpolation="none",cmap="gray")
        axis("off");  title(L"$x\sim p_{\theta}(x)$")

        subplot(4,5,2); imshow(X_gen[:,:,1,2], aspect=1, vmin=0,vmax=1,interpolation="none",  cmap="gray")
        axis("off"); title(L"$x\sim p_{\theta}(x)$")

        subplot(4,5,3); imshow(X_train_latent[:,:,1,1] |> cpu, interpolation="none", cmap="gray")
        axis("off"); title(L"$x_{train1} \sim p(x)$")

        subplot(4,5,4); imshow(ZX_train_sq[:,:,1,1],interpolation="none", vmin=-3, vmax=3, cmap="seismic"); axis("off"); 
        title(L"$z_{train1} = G^{-1}(x_{train1})$ "*string("\n")*" mean "*string(mean_train_1)*" std "*string(std_train_1));
        
        subplot(4,5,6); imshow(X_gen[:,:,1,3], vmin=0,vmax=1,interpolation="none",  cmap="gray")
        axis("off");  title(L"$x\sim p_{\theta}(x)$")

        subplot(4,5,7); imshow(X_gen[:,:,1,4], vmin=0,vmax=1, interpolation="none", cmap="gray")
        axis("off"); title(L"$x\sim p_{\theta}(x)$")

        subplot(4,5,8); imshow(X_train_latent[:,:,1,2]|> cpu, aspect=1, interpolation="none",  cmap="gray")
        axis("off"); title(L"$x_{train2} \sim p(x)$")

          
        subplot(4,5,9) ;imshow(ZX_train_sq[:,:,1,2],   interpolation="none", 
                                        vmin=-3, vmax=3, cmap="seismic"); axis("off"); 
        title(L"$z_{train2} = G^{-1}(x_{train2})$ "*string("\n")*" mean "*string(mean_train_2)*" std "*string(std_train_2));
        subplot(4,5,11); imshow(X_gen[:,:,1,5], vmin=0,vmax=1,interpolation="none", cmap="gray")
        axis("off");  title(L"$x\sim p_{\theta}(x)$")

        subplot(4,5,12); imshow(X_gen[:,:,1,6],  vmin=0,vmax=1, interpolation="none", cmap="gray")
        axis("off"); title(L"$x\sim p_{\theta}(x)$")

        subplot(4,5,13); imshow(X_test_latent[:,:,1,1]|> cpu,  interpolation="none",  cmap="gray")
        axis("off"); title(L"$x_{test1} \sim p(x)$")

        subplot(4,5,14) ;imshow(ZX_test_sq[:,:,1,1],  interpolation="none", 
                                        vmin=-3, vmax=3, cmap="seismic"); axis("off"); 
        title(L"$z_{test1} = G^{-1}(x_{test1})$ "*string("\n")*" mean "*string(mean_test_1)*" std "*string(std_test_1));
            

        subplot(4,5,16); imshow(X_gen[:,:,1,7], vmin=0, vmax=1, interpolation="none", cmap="gray"); 
        axis("off");  title(L"$x\sim p_{\theta}(x)$")

        subplot(4,5,17); imshow(X_gen[:,:,1,8], vmin=0,vmax=1, interpolation="none",  cmap="gray")
        axis("off"); title(L"$x\sim p_{\theta}(x)$")

        subplot(4,5,18); imshow(X_test_latent[:,:,1,2]|> cpu,  interpolation="none", cmap="gray")
        axis("off"); title(L"$x_{test2} \sim p(x)$")

          
        subplot(4,5,19); imshow(ZX_test_sq[:,:,1,2], interpolation="none", vmin=-3, vmax=3, cmap="seismic");
        axis("off"); title(L"$z_{test2} = G^{-1}(x_{test2})$ "*string("\n")*" mean "*string(mean_test_2)*" std "*string(std_test_2));
        tight_layout()

        fig_name = @strdict e λ lr lr_step noiseLev n_hidden L K num_digit total_seen
        safesave(joinpath(save_path, savename(fig_name; digits=6)*"_glow_latent.png"), fig); close(fig)
      

        ############# Training metric logs
        vfloss   = vec(floss)
        vflogdet = vec(flogdet)
        vsum = vfloss + vflogdet

        vfloss_test   = vec(floss_test)
        vflogdet_test = vec(flogdet_test)
        vsum_test = vfloss_test + vflogdet_test

        fig = figure("training logs ", figsize=(10,8))
        vfloss_epoch = vfloss[1:findall(x -> x == 0,vfloss)[1]-1]
        vfloss_epoch_test = vfloss_test[1:findall(x -> x == 0,vfloss_test)[1]-1]

        subplot(3,1,1)
        title("L2 Term: train="*string(vfloss_epoch[end])*" test="*string(vfloss_epoch_test[end]))
        plot(vfloss_epoch, label="train");
        plot(1:nbatches:nbatches*e, vfloss_epoch_test, label="test"); 
        axhline(y=1f0,color="red",linestyle="--",label="Noise Likelihood")
        xlabel("Parameter Update"); legend()
        
        subplot(3,1,2)
        vflogdet_epoch = vflogdet[1:findall(x -> x == 0,vflogdet)[1]-1]
        vflogdet_epoch_test = vflogdet_test[1:findall(x -> x == 0,vflogdet_test)[1]-1]
        title("Logdet Term: train="*string(vflogdet_epoch[end])*" test="*string(vflogdet_epoch_test[end]))
        plot(vflogdet_epoch);
        plot(1:nbatches:nbatches*e, vflogdet_epoch_test);
        xlabel("Parameter Update")  

        subplot(3,1,3)
        vsum_epoch = vsum[1:findall(x -> x == 0,vsum)[1]-1]
        vsum_epoch_test = vsum_test[1:findall(x -> x == 0,vsum_test)[1]-1]
        plot(vsum_epoch); title("Total Objective: train="*string(vsum_epoch[end])*" test="*string(vsum_epoch_test[end]))
        plot(1:nbatches:nbatches*e, vsum_epoch_test); 
        xlabel("Parameter Update") 

        tight_layout()

        fig_name = @strdict nepochs e lr lr_step λ noiseLev n_hidden L K num_digit total_seen
        safesave(joinpath(save_path, savename(fig_name; digits=6)*"mnist_glow_log.png"), fig); close(fig)
      
    end

    #save params every 4 epochs
    # if(mod(e,intermediate_save_params)==0) 
    #      # Saving parameters and logs
    #      Params = get_params(G) |> cpu 
    #      save_dict = @strdict e nepochs lr lr_step λ noiseLev n_hidden L K  Params floss flogdet num_digit total_seen
    #      @tagsave(
    #          datadir(sim_name, savename(save_dict, "jld2"; digits=6)),
    #          save_dict;
    #          safe=true
    #      )
    # end

end

