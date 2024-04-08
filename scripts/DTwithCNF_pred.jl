using DrWatson
@quickactivate "Domain_transfer"

using Flux
using LinearAlgebra
using Random
# using Metalhead
using JLD2
using Statistics
using ImageQualityIndexes 
using PyPlot
using SlimPlotting
using InvertibleNetworks
using MLUtils
# using BSON 
# using Wavelets
using StatsBase
using Distributions
using Images
using UNet

#### DATA LOADING #####
nx,ny = 128, 128
N = nx*ny;


data_path= "../data/CompassShotmid.jld2"
# datadir(CompassShot.jld2)
train_X = jldopen(data_path, "r")["X"]
train_y = jldopen(data_path, "r")["Y"]
  
train_xA = zeros(Float32, nx, ny, 1,8)
train_xB = zeros(Float32, nx, ny, 1,8)

img=2004
  
for i=1:8
    sigma = 1.0
    train_xA[:,:,:,i] = imresize(imfilter(train_X[:,:,img+i],KernelFactors.gaussian((sigma,sigma))),(nx,ny))
    train_xB[:,:,:,i] = imresize(imfilter(train_X[:,:,2160+img+i],KernelFactors.gaussian((sigma,sigma))),(nx,ny))
end


train_xA = train_xA.* (abs.(train_xA) .> 5e-6)
train_xB = train_xB.* (abs.(train_xB) .> 5e-6)

maxa=maximum(abs.(train_xA))
maxb=maximum(abs.(train_xB))

train_xA = (train_xA )./(maxa) 
train_xB = (train_xB )./(maxb) 

# Define the generator and discriminator networks

path = "/home/ykartha6/juliacode/Domain_transfer/Bestresults/K=10_L=3_e=310_lr=5e-5_n_hidden=512.jld2"
function get_network(path)
  # test parameters
  batch_size = 2
  n_post_samples = 128
  device = gpu
  #load previous hyperparameters
  bson_file = JLD2.load(path);
  n_hidden = bson_file["n_hidden"];
  L = bson_file["L"];
  K = bson_file["K"];
  Params = bson_file["Params"];
  e = bson_file["e"];
  G = NetworkConditionalGlow(1, 1, n_hidden,  L, K; split_scales=true, activation=SigmoidLayer(low=0.5f0,high=1.0f0));
  p_curr = get_params(G);
  for p in 1:length(p_curr)
  p_curr[p].data = Params[p].data
  end
  return G
end

G = get_network(path)
generator = G |> gpu
device=gpu
imgs = 8
n_train = 8

n_batches = cld(8,8)
YA = ones(Float32,nx,ny,1,imgs) + randn(Float32,nx,ny,1,imgs) ./1000
YB = ones(Float32,nx,ny,1,imgs) .*7 + randn(Float32,nx,ny,1,imgs) ./1000



idx_eA = reshape(randperm(n_train), 8, n_batches)
idx_eB = reshape(randperm(n_train), 8, n_batches)

XA = train_xA[:, :, :,:] + randn(Float32,(nx,ny,1,8)) ./1f5
XB = train_xB[:, :, :,:] + randn(Float32,(nx,ny,1,8)) ./1f5

X = cat(XA, XB,dims=4)
Y = cat(YA, YB,dims=4)

Zx, Zy, lgdet = generator.forward(X|> device, Y|> device)  #### concat so that network normalizes ####

ZyA = Zy[:,:,:,1:imgs]
ZyB = Zy[:,:,:,imgs+1:end]

ZxA = Zx[:,:,:,1:imgs]
ZxB = Zx[:,:,:,imgs+1:end]

Zy1 = cat(ZyB,ZyA,dims=4)


fake_images,invcall = generator.inverse(Zx|>device,Zy1)  ###### generating images #######
 
fake_imagesAfromB = fake_images[:,:,:,imgs+1:end]
fake_imagesBfromA = fake_images[:,:,:,1:imgs]

# fake_imagesAfromBavg=  zeros(Float32,(nx,ny,1,1))
# fake_imagesBfromAavg=  zeros(Float32,(nx,ny,1,1))
# for i=1:32
#   a = fake_imagesAfromB[:,:,1,i]|>cpu
#   fake_imagesAfromBavg +=a

#   b = fake_imagesBfromA[:,:,1,i]|>cpu
#   fake_imagesBfromAavg +=b
# end

# fake_imagesAfromBavg = fake_imagesAfromBavg./32
# fake_imagesBfromAavg = fake_imagesBfromAavg./32

XB = XB.*maxb
XA = XA.*maxa

fake_imagesBfromA = fake_imagesBfromA.*maxb
fake_imagesAfromB = fake_imagesAfromB.*maxa

e=1

          ####### discrim training ########
    plot_sdata(XB[:,:,1,1]|>cpu,(14.06,4.976),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel+den ")
    plt.savefig("../plots/Shot_rec_df/vel+den data test1.png")
    plt.close()

    plot_sdata(fake_imagesBfromA[:,:,1,1]|>cpu,(14.06,4.976),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel+den from vel 1_$e ")
    plt.savefig("../plots/Shot_rec_df/vel+den test pred1_$e.png")
    plt.close()

    plot_sdata(XA[:,:,1,1]|>cpu,(14.06,4.976),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel ")
    plt.savefig("../plots/Shot_rec_df/vel data test1.png")
    plt.close()

    plot_sdata(fake_imagesAfromB[:,:,1,1]|>cpu,(14.06,4.976),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel from vel+den 1_$e ")
    plt.savefig("../plots/Shot_rec_df/vel test pred1_$e.png")
    plt.close()

    plot_sdata((XB[:,:,1,1]|>cpu)-(fake_imagesAfromB[:,:,1,1]|>cpu),(14.06,4.976),cbar=true)
    plt.title("difference in Xb and fakeAB ")
    plt.savefig("../plots/Shot_rec_df/vel+dendiff.png")
    plt.close()

    plot_sdata((XB[:,:,1,1]|>cpu)-(fake_imagesBfromA[:,:,1,1]|>cpu),(14.06,4.976),cbar=true)
    plt.title.(" difference in fake velden and domain_transfe_velden ")
    plt.savefig("../plots/Shot_rec_df/veldendiff_$e.png")
    plt.close()

    plt.plot((fake_imagesBfromA[20:end,64,1,1]|>cpu),label="veldenpredicted")
    plt.plot((XB[20:end,64,1,1]|>cpu),label="truevelden")
    plt.plot((XA[20:end,64,1,1]|>cpu),label="truevel")
    plt.title.(" difference in fakeB and XB trace ")
    plt.legend()
    plt.savefig("../plots/Shot_rec_df/veldifftrace$e.png")    
    plt.close()