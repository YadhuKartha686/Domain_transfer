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

#### DATA LOADING #####

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

batch_size = 8
nx,ny = 2048, 512
N = nx*ny;

n_train = 1504

n_batches = cld(n_train,batch_size)


# Define the generator and discriminator networks

n_epochs     = 100
device = gpu
lr = 1f-4
lr_step   = 10
lr_rate = 0.75f0
clipnorm_val = 10f0
noise_lev_x  = 0.005f0
noise_lev_y  = 0.00f0
split_scales = true

#User params
save_every   = 5
plot_every   = 1

# Number of samples to test conditional mean quality metric on.
n_condmean = 32
posterior_samples = 32 


K = 9
L = 5# l=5 is better
n_hidden = 64
low = 0.5f0
n_in = 1
n_out = 1
G = NetworkConditionalGlow(n_out, n_in, n_hidden,  L, K; split_scales=split_scales, activation=SigmoidLayer(low=low,high=1.0f0));


model = Chain(
    Chain([
      Conv((7, 7), 1 => 64, pad=3, stride=2, bias=false),  # 9_408 parameters
      BatchNorm(64, relu),            # 128 parameters, plus 128
      MaxPool((3, 3), pad=1, stride=2),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64, relu),        # 128 parameters, plus 128
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64),              # 128 parameters, plus 128
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64, relu),        # 128 parameters, plus 128
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64),              # 128 parameters, plus 128
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 128, pad=1, stride=2, bias=false),  # 73_728 parameters
          BatchNorm(128, relu),       # 256 parameters, plus 256
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128),             # 256 parameters, plus 256
        ),
        Chain([
          Conv((1, 1), 64 => 128, stride=2, bias=false),  # 8_192 parameters
          BatchNorm(128),             # 256 parameters, plus 256
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128, relu),       # 256 parameters, plus 256
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128),             # 256 parameters, plus 256
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 128 => 256, pad=1, stride=2, bias=false),  # 294_912 parameters
          BatchNorm(256, relu),       # 512 parameters, plus 512
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256),             # 512 parameters, plus 512
        ),
        Chain([
          Conv((1, 1), 128 => 256, stride=2, bias=false),  # 32_768 parameters
          BatchNorm(256),             # 512 parameters, plus 512
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256, relu),       # 512 parameters, plus 512
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256),             # 512 parameters, plus 512
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 256 => 512, pad=1, stride=2, bias=false),  # 1_179_648 parameters
          BatchNorm(512, relu),       # 1_024 parameters, plus 1_024
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512),             # 1_024 parameters, plus 1_024
        ),
        Chain([
          Conv((1, 1), 256 => 512, stride=2, bias=false),  # 131_072 parameters
          BatchNorm(512),             # 1_024 parameters, plus 1_024
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512, relu),       # 1_024 parameters, plus 1_024
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512),             # 1_024 parameters, plus 1_024
        ),
        identity,
      ),
    ]),
    Chain(
      AdaptiveMeanPool((1, 1)),
      MLUtils.flatten,
      Dense(512 => 2, sigmoid),             # 513_000 parameters
    ),
  )



# Define the loss functions
function Dissloss(A, Afake)
    real_loss = -mean(log.(A))
    fake_loss = -mean(log.(1.0 .- Afake))
    return real_loss + fake_loss
end

function Genloss(Afake)
    fake_loss = -mean(log.(1.0 .- Afake))
    return fake_loss
end


# Initialize networks and optimizers
generator = G |> gpu
discriminatorA = gpu(model)
discriminatorB = gpu(model)

generator_params = Flux.params(generator)
discriminatorA_params = Flux.params(discriminatorA)
discriminatorB_params = Flux.params(discriminatorA)

opt_adam = "adam"
optimizer_g = Flux.Optimiser(ExpDecay(lr, lr_rate, n_batches*lr_step, 1f-6), ClipNorm(clipnorm_val), ADAM(lr))
optimizer_da = Flux.Optimiser(ExpDecay(lr, lr_rate, n_batches*lr_step, 1f-6), ClipNorm(clipnorm_val), ADAM(lr))
optimizer_db = Flux.Optimiser(ExpDecay(lr, lr_rate, n_batches*lr_step, 1f-6), ClipNorm(clipnorm_val), ADAM(lr))

# Main training loop
for e=1:n_epochs# epoch loop
    idx_e = reshape(randperm(n_train), batch_size, n_batches)
    for b = 1:n_batches # batch loop
    	@time begin
	        X = train_x1[:, :, :, idx_e[:,b]];
	        Y = train_Y[:, :, :, idx_e[:,b]];
	        X .+= noise_lev_x*randn(Float32, size(X));
            Y .+= 0.00001*randn(Float32, size(Y));
        
        
            # Compute discriminator loss
            ZxB, ZyB, lgdetb = generator.forward(XB|> device, YB|> device)
            ZxA, ZyA, lgdeta = generator.forward(XA|> device, YA|> device)


            fake_imagesAfromB = generator.inverse(ZxB,ZyA)
            dA_grads = Flux.gradient(discriminatorA_params) do
                real_outputA = discriminatorA(XA|> device)
                fake_outputA = discriminatorA(fake_imagesAfromB)
                lossA = Dissloss(real_outputA, fake_outputA)
            end
            Flux.Optimise.update!(optimizer_da, discriminatorA_params,dA_grads)

            fake_imagesBfromA = generator.inverse(ZxA,ZyB)
            dB_grads = Flux.gradient(discriminatorB_params) do
                real_outputB = discriminatorB(XB|> device)
                fake_outputB = discriminatorB(fake_imagesBfromA)
                lossB = Dissloss(real_outputB, fake_outputB)
            end
            Flux.Optimise.update!(optimizer_db, discriminatorB_params,dB_grads)

            # Compute generator loss
            g_grads = Flux.gradient(generator_params) do
                fake_outputA = discriminatorA(fake_imagesAfromB)
                fake_outputB = discriminatorB(fake_imagesBfromA)
                lossA = Genloss(fake_outputA)
                lossB = Genloss(fake_outputB)
                ml = norm(ZxA)^2 -lgdeta / N + norm(ZxB)^2 -lgdetb / N 
                loss = lossA+lossB+ml
            end
            Flux.Optimise.update!(optimizer_g, generator_params,g_grads)


            #loss calculation for printing




        end    
    end
    
    if epoch % 100 == 0
        println("Epoch: $epoch, Generator Loss: $g_loss, Discriminator Loss: $d_loss")
        
        # Optionally, generate and save a sample image during training
        sample_noise = randn(latent_dim, 1)
        generated_image = generator(sample_noise)
        # Save or visualize the generated_image as needed
    end
end
