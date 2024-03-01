using DrWatson
@quickactivate "Domain_transfer"


using JLD2

using PyPlot
using SlimPlotting

using Images


#### DATA LOADING #####
nx,ny = 128, 128
N = nx*ny;

data_path= "../data/CompassShotmid.jld2"
# datadir(CompassShot.jld2)
train_X = jldopen(data_path, "r")["X"]
train_y = jldopen(data_path, "r")["Y"]
  
train_xA = zeros(Float32, nx, ny, 1,2160)
train_xB = zeros(Float32, nx, ny, 1,2160)

  
for i=1:2160
    sigma = 1.0
    train_xA[:,:,:,i] = imresize(imfilter(train_X[:,:,i],KernelFactors.gaussian((sigma,sigma))),(nx,ny))
    train_xB[:,:,:,i] = imresize(imfilter(train_X[:,:,2160+i],KernelFactors.gaussian((sigma,sigma))),(nx,ny))
end


plot_sdata(train_xB[:,:,1,1],(7.03,2.488),perc=95,vmax=0.03,cbar=true)
plt.savefig("data test2.png")
plt.close()
plt.title("data test vel+den ")