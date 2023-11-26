using DrWatson
@quickactivate "Domain_transfer"

using Flux
using LinearAlgebra
using Random

using JLD2
using Statistics
using ImageQualityIndexes 
using PyPlot
using SlimPlotting
using InvertibleNetworks
# using UNet
using BSON 
# using Wavelets
using StatsBase
using Distributions
using Images
matplotlib.use("Agg");
plot_path = plotsdir();

Random.seed!(1234)
# Y_train, X_train, Y_val, X_val, Y_test, X_test = get_data_train_64_gen()

function posterior_sampler(G, y, size_x; device=gpu, num_samples=1, batch_size=16)
    # make samples from posterior for train sample
	X_forward = randn(Float32, size_x[1:end-1]...,batch_size) |> device
    Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
    _, Zy_fixed_train, _ = G.forward(X_forward, Y_train_latent_repeat); #needs to set the proper sizes here
    X_post_train = zeros(Float32, size_x[1:end-1]...,num_samples)
    for i in 1:div(num_samples, batch_size)
    	ZX_noise_i = randn(Float32, size_x[1:end-1]...,batch_size)|> device
   		X_post_train[:,:,:, (i-1)*batch_size+1 : i*batch_size] = G.inverse(
        	ZX_noise_i,
        	Zy_fixed_train
    		)[1] |> cpu;
	end
	X_post_train
end



function normalize_images(data)
    num_images, height, width, channels = size(data)
    
    for i in 1:num_images
        min_vals = minimum(data[:, :, :, i])
        max_vals = maximum(data[:, :, :, i])
        data[:, :, :, i] .= (data[:, :, :, i] .- min_vals) ./ (max_vals .- min_vals)
    end
    
    return data
end
  
  
  #Loading Data
data_path= "../data/CompassShot.jld2"
# datadir(CompassShot.jld2)
train_X = jldopen(data_path, "r")["X"]
train_y = jldopen(data_path, "r")["Y"]
  
test_x1 = zeros(Float32, 2048, 512, 1,301)
  
for i=1:301
    sigma = 1.0
    test_x1[:,:,:,i] = imresize(imfilter(train_X[:,:,i+1499],KernelFactors.gaussian((sigma,sigma))),(2048,512))
end
test_y = train_y[1500:end,:]
  
  
train_x1 = zeros(Float32, 2048, 512, 1,1504)
  
for i=1:1504
    sigma = 1.0
    train_x1[:,:,:,i] = imresize(imfilter(train_X[:,:,i],KernelFactors.gaussian((sigma,sigma))),(2048,512))
end
  
train_y = train_y[1:1504,:]
train_Y = ones(Float32,2048,512,1,1504)
for i=1:1504
    if train_y[i,1]==1.0
        train_Y[:,:,:,i]= train_Y[:,:,:,i] .*2
    end
end

test_Y = ones(Float32,2048,512,1,300)
for i=1:300
    if test_y[i,1]==1.0
        test_Y[:,:,:,i]=test_Y[:,:,:,i] .* 2
    end
end


# Choose params
batch_size = 8
nx,ny = 2048, 512
N = nx*ny;

n_train = 1504

n_batches = cld(n_train,batch_size)

# Training hyperparameters 
n_epochs     = 150
device = gpu
lr = 1f-4
lr_step   = 10
lr_rate = 0.75f0
clipnorm_val = 10f0
noise_lev_x  = 0.005f0
noise_lev_y  = randn(Float32,(2048,512,1,batch_size))./1000
split_scales = true

#User params
save_every   = 5
plot_every   = 1

# Number of samples to test conditional mean quality metric on.
n_condmean = 32
posterior_samples = 32 

# Create conditional network
K = 9
L = 5# l=5 is better
n_hidden = 64
low = 0.5f0
n_in = 1
n_out = 1
#Choose model
G = NetworkConditionalGlow(n_out, n_in, n_hidden,  L, K; split_scales=split_scales, activation=SigmoidLayer(low=low,high=1.0f0));
G = G |> device;

# Optimizer
opt_adam = "adam"
opt = Flux.Optimiser(ExpDecay(lr, lr_rate, n_batches*lr_step, 1f-6), ClipNorm(clipnorm_val), ADAM(lr))

# Training logs 
loss   = [];
logdet_train = [];
mseval = [];
ssim   = [];
l2_cm  = [];

loss_val   = [];
logdet_val = [];
ssim_val   = [];
l2_cm_val  = [];

vmax_val = 200
vmin_val = -125
num_post_samples=32
lat_vmax = 8
lat_vmin = -8

Ztest = randn(Float32, nx,ny,1,batch_size); 

#Training
function sigmoid_schedule(t , T , tau =0.6 , start =0 , end_t =3 , clip_min =1f-9) 
    # A scheduling function based on sigmoid function with a temperature tau .
        v_start = sigmoid( start / tau )
        v_end = sigmoid( end_t / tau )
        return ( v_end - sigmoid(( t /T * ( end_t - start ) + start ) / tau )) / ( v_end - v_start )
    end
    tau = 0.6

for e=1:n_epochs# epoch loop
    # sigma_noise = sigmoid_schedule(e,n_epochs,tau)
    # alpha_noise = sqrt(1-sigma_noise^2)
    idx_e = reshape(randperm(n_train), batch_size, n_batches)
    for b = 1:n_batches # batch loop

    	@time begin
	        X = train_x1[:, :, :, idx_e[:,b]];
	        Y = train_Y[:, :, :, idx_e[:,b]];
	        X .+= noise_lev_x*randn(Float32, size(X));
            # X = alpha_noise .* X + sigma_noise*randn(Float32, size(X));

			Y = Y + noise_lev_y;
      
	        # Forward pass of normalizing flow
	        Zx, Zy, lgdet = G.forward(X|> device, Y|> device)

            

            fakeimgs,invcall = G.inverse(Ztest|> device,Zy)

	        # Loss function is l2 norm 
	        append!(loss, norm(Zx)^2 / (N*batch_size))  # normalize by image size and batch size
	        append!(logdet_train, -lgdet / N) # logdet is internally normalized by batch size

            grad_fake_images = gradient(x -> Flux.mse(X|> device,x), fakeimgs)[1]
            G.backward_inv(grad_fake_images/batch_size, fakeimgs, invcall;)

            mseloss = Flux.mse(X|> device,fakeimgs)
            append!(mseval, mseloss)

            for p in get_params(G)
                Flux.update!(opt,p.data,p.grad)
            end
            clear_grad!(G)

	        println("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
	            "; f l2 = ",  loss[end], 
                "; lgdet = ", logdet_train[end], "; f = ", loss[end] + logdet_train[end], "; mse = ", mseloss, "\n")
            

	        Base.flush(Base.stdout)
            plt.plot(loss)
            plt.title("loss $b")
            plt.savefig("../plots/Shot_rec/loss$e.png")
            plt.close()
            plt.plot(mseval)
            plt.title("mseloss $b")
            plt.savefig("../plots/Shot_rec/mseloss$e.png")
            plt.close()
            plt.plot(logdet_train)
            plt.title("logdet $b")
            plt.savefig("../plots/Shot_rec/logdet$e.png")
            plt.close()
    	end
    end

    if(mod(e,plot_every)==0) 
        for (test_x, test_y, file_str) in [[train_x1,train_Y, "train"], [test_x1, test_Y, "test"]]
            num_cols = 8
            plots_len = 2
            all_sampls = size(test_x)[end]
            fig = figure(figsize=(25, 5)); 
            for (i,ind) in enumerate((1:div(all_sampls,3):all_sampls)[1:plots_len])
                x = test_x[:,:,:,ind:ind] 
                y = test_y[:,:,:,ind:ind]
                y .+= randn(Float32).*randn(Float32, size(y))./1000;
        
                # make samples from posterior for train sample 
                X_post = posterior_sampler(G,  y, size(x); device=device, num_samples=num_post_samples,batch_size)|> cpu
                X_post_mean = mean(X_post,dims=4)
                X_post_std  = std(X_post, dims=4)
        
                x_hat = X_post_mean[:,:,1,1]
                x_gt =  x[:,:,1,1]
                error_mean = abs.(x_hat-x_gt)
        
                ssim_i = round(assess_ssim(x_hat, x_gt),digits=2)
                rmse_i = round(sqrt(mean(error_mean.^2)),digits=4)
        
                y_plot = y[:,:,1,1]'
                a = maximum(x_gt)
                b = minimum(x_gt)
        
                # subplot(plots_len,num_cols,(i-1)*num_cols+1); imshow(y_plot, vmin=-a,vmax=a,interpolation="none", cmap="gray")
                # axis("off"); title(L"rtm");colorbar(fraction=0.046, pad=0.04);
        
        
                subplot(plots_len,num_cols,(i-1)*num_cols+3); imshow(X_post[:,:,1,1],vmin=b,vmax=a, interpolation="none", cmap="gray")
                axis("off"); title("Posterior sample"); colorbar(fraction=0.046, pad=0.04);
        
                subplot(plots_len,num_cols,(i-1)*num_cols+4); imshow(X_post[:,:,1,2], vmin=b,vmax=a,interpolation="none", cmap="gray")
                axis("off");title("Posterior sample") ; colorbar(fraction=0.046, pad=0.04);title("Posterior sample")
        
                subplot(plots_len,num_cols,(i-1)*num_cols+5); imshow(x_gt, vmin=b,vmax=a,   interpolation="none", cmap="gray")
                axis("off"); title(L"Reference $\mathbf{x^{*}}$") ; colorbar(fraction=0.046, pad=0.04);
        
                subplot(plots_len,num_cols,(i-1)*num_cols+6); imshow(x_hat , vmin=b,vmax=a,   interpolation="none", cmap="gray")
                axis("off"); title("Posterior mean | SSIM="*string(ssim_i)) ; colorbar(fraction=0.046, pad=0.04);
        
                subplot(plots_len,num_cols,(i-1)*num_cols+7); imshow(error_mean , vmin=0,vmax=nothing, interpolation="none", cmap="gray")
                axis("off");title("Error | RMSE="*string(rmse_i)) ; cb = colorbar(fraction=0.046, pad=0.04);
        
                subplot(plots_len,num_cols,(i-1)*num_cols+8); imshow(X_post_std[:,:,1,1] , vmin=0,vmax=nothing,interpolation="none", cmap="gray")
                axis("off"); title("Standard deviation") ;cb =colorbar(fraction=0.046, pad=0.04);

        end
    
        tight_layout()
        fig_name = @strdict e lr n_hidden L K batch_size
        safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_"*file_str*".png"), fig); close(fig)

        
        XAD = train_x1[:,:,:,1:8]
        # XA = train_x1[:,:,:,1:8]

        YAD = train_Y[:,:,:,1:8]
        YAD = YAD + noise_lev_y
        # YA = train_Y[:,:,:,4:5]

        shot_rec = zeros(Float32,2048,512,1,8)

        _, Zy_fixed_train, _ = G.forward(XAD |> device, YAD |> device); #needs to set the proper sizes here
        ZX_noise_i = randn(Float32, 2048,512,1,8)|> device
        shot_rec[:,:,:, 1:8] = G.inverse( ZX_noise_i,Zy_fixed_train)[1] |> cpu;

        # _, Zy_fixed_train, _ = G.forward(XA |> device, YA |> device); #needs to set the proper sizes here
        # ZX_noise_i = randn(Float32, 2048,512,1,2)|> device
        # shot_rec[:,:,:,3:4] = G.inverse( ZX_noise_i,Zy_fixed_train)[1] |> cpu;

        plot_sdata(XAD[:,:,:,1],(0.8,1),vmax=0.04f0,perc=95,cbar=true)
        plt.title("Shot record train ( vel + den) $e")
        plt.savefig("../plots/Shot_rec/vel+den train$e.png")
        plt.close()

        plot_sdata(shot_rec[:,:,:,1],(0.8,1),vmax=0.04f0,perc=95,cbar=true)
        plt.title("Shot record pred ( vel + den) $e")
        plt.savefig("../plots/Shot_rec/vel+den$e.png")
        plt.close()

        plot_sdata(shot_rec[:,:,:,4],(0.8,1),vmax=0.04f0,perc=95,cbar=true)
        plt.title("Shot record pred ( vel) $e")
        plt.savefig("../plots/Shot_rec/vel$e.png")
        plt.close()


        # sum = loss + logdet
		# sum_test = loss_test + logdet_test

		# fig1 = figure(figsize=(20,20))
		# subplot(3,1,1); title("L2 Term: train="*string(loss[end])*" test="*string(loss_test[end]))
		# plot(loss, label="train");
		# plot(n_batches:n_batches:n_batches*e, loss_test, label="test"); 
		# axhline(y=1,color="red",linestyle="--",label="Normal Noise")
		# xlabel("Parameter Update"); legend();
		# ylim(0,2)

		# subplot(3,1,2); title("Logdet Term: train="*string(logdet[end])*" test="*string(logdet_test[end]))
		# plot(logdet);
		# plot(n_batches:n_batches:n_batches*e, logdet_test);
		# xlabel("Parameter Update") ;

		# subplot(3,1,3); title("Total Objective: train="*string(sum[end])*" test="*string(sum_test[end]))
		# plot(sum); 
		# plot(n_batches:n_batches:n_batches*e, sum_test); 
		# xlabel("Parameter Update") ;

        # tight_layout()
		# fig_name = @strdict freeze_conv opt_adam clip_norm  e n_epochs n_train  lr lr_rate lr_step noise_lev n_hidden L K batch_size
		# safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_log.png"), fig1); close(fig1)
    end


	end


end

println("... train_cond_64_gen Done! ...") 

#include("/home/ykartha6/juliacode/Conditional NF on leak no leak.jl")