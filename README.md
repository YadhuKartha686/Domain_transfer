# Domain_transfer

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> Domain_transfer

It is authored by YadhuMK.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "Domain_transfer"
```
which auto-activate the project and enable local path handling from DrWatson.


```
scp -r ykartha6@cruyff.cc.gatech.edu:/nethome/ykartha6/Domain_transfer/plots Domain_transfer/plots 
module load Julia
salloc -A ykartha6 -t 20:00:00 --cpus-per-task=16 --mem-per-cpu=16G --gres=gpu:1 --ntasks=1 srun --pty julia
squeue to check if im safe
scanel with job id if didnt use srun --pty julia


for sbatch

use test-julia_slurm.sh file and set parametes
sbatch test-julia_slurm.sh

```