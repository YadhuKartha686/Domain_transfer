using DrWatson
@quickactivate "Domain_transfer"

using Flux
using LinearAlgebra
using Random
using Metalhead
using JLD2
using Statistics
using ImageQualityIndexes 
using PyPlot
using SlimPlotting
using InvertibleNetworks
using MLUtils
using BSON 
# using Wavelets
using StatsBase
using Distributions
using Images
using MLDatasets
plot_path = "."

using InvertibleNetworks, Flux
using PyPlot
using LinearAlgebra, Random
using MLDatasets

plot_path = "."

# Training hyperparameters
device = cpu #GPU does not accelerate at this small size. quicker on cpu
lr     = 4f-3
epochs = 30
batch_size = 1

# Load in training data
n_total = 10
X, train_y = MNIST(split=:train)[1:n_total];
X, train_y = MNIST.traindata#[1:n_total]
train_x, _ = MNIST.traindata()
X = train_x[:,:,1:n_total]

# train_x1 = normalize_images(train_x1)


# Architecture parametrs
chan_x = 1; chan_y = 1; L = 2; K = 10; n_hidden = 32 # Number of hidden channels in convolutional residual blocks

# Create network
G = NetworkConditionalGlow(chan_x, chan_y, n_hidden,  L, K; split_scales=true) |> device;
opt = Flux.ADAM(lr)

factor = 1f-20

x_gt = train_x1[:,:,:,1:1]
Z_fix =  randn(Float32,size(x_gt))
y = randn(Float32,size(x_gt))

loss      = []; logdet_train = []; 
n_epochs = 5000
#pretrain to output water. 
for e=1:n_epochs # epoch loop
	Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
	_, Zy_fixed_train, lgdet = G.forward(Z_fix|>device, Y_train_latent_repeat); #needs to set the proper sizes here

	X_gen, Y_gen  = G.inverse(Z_fix|>device,Zy_fixed_train);
	X_gen_cpu = X_gen |>cpu

	# gs = zeros(Float32,nx,ny,1,batch_size);
	f_all = 0
	for i in 1:batch_size
		g = (X_gen_cpu[:,:,1,i] .- x_gt)  
		f = norm(g)^2
		# gs[:,:,:,i] =  g
		f_all += f
	end
	gs = gradient(x -> Flux.mse(x_gt|> device,x), X_gen)[1]

	if mod(e,20) == 0
		gs = gs|> cpu
	    plot_sdata(x_gt[:,:,:,1],(0.8,1),vmax=0.04f0,perc=95,cbar=true)
        plt.title("Shot record train ( vel + den) $e")
        plt.savefig("../plots/Shot_rec/vel+den train$e.png")
        plt.close()

        plot_sdata(X_gen[:,:,:,1]|>cpu,(0.8,1),vmax=0.04f0,perc=95,cbar=true)
        plt.title("Shot record pred ( vel + den) $e")
        plt.savefig("../plots/Shot_rec/vel+den$e.png")
        plt.close()
	end

	# Loss function is l2 norm 
	append!(loss, f_all / batch_size)  # normalize by image size and batch size
	append!(logdet_train, -lgdet / N) # logdet is internally normalized by batch size

	# Set gradients of flow and summary network
	ΔX, X, ΔY = G.backward_inv(((gs ./ factor)|>device) / batch_size, X_gen, Y_gen)

	for p in get_params(G) 
		Flux.update!(opt,p.data,p.grad)	
	end; clear_grad!(G)

	print("Iter: epoch=", e, "/", n_epochs, 
	    "; f l2 = ",  loss[end], 
	    "; lgdet = ", logdet_train[end], "; f = ", loss[end] + logdet_train[end], "\n")
	Base.flush(Base.stdout)
end