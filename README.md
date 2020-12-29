This readme has info on setting up Dreamcoder and working with our codebase. For
more details on using Dreamcoder, you can look at readme inside the `ec` directory.

# How to start using Dreamcoder to solve ARC

# Setup

## Clone this repo

Ideally, clone it to openmind…
```
ssh [username]@openmind7.mit.edu
cd /om2/user/[username]
git clone https://github.com/anshula/neurosymbolic-modules.git
```

...but if you don’t have access to openmind, clone it to another linux machine.  The machine probably needs to be linux in order for singularity to run on it smoothly.

## Clone the submodules of this repo

First make sure you are in the `neurosymbolic-modules` folder, then:

```
cd ec
git clone https://github.com/insperatum/pinn.git
cd pinn
git checkout 1878ef5

cd ..
git clone https://github.com/insperatum/pregex.git
cd pregex
git checkout b5eab11

cd ..
git clone https://github.com/hans/pyccg.git
cd pyccg
git checkout c465a23
```

## Add the singularity container

Download the singularity `container.img` file, and put it directly inside the `neurosymbolic modules/ec` folder.
- Quick way (works on openmind): copy it from Anshula’s openmind folder: `srun cp /cbcl/cbcl01/anshula/shared/container.img /om2/user/$USER/neurosymbolic-modules/ec/container.img`
- Another quick way (works without openmind): copy it from Anshula's dropbox; `wget "https://www.dropbox.com/s/5ufvzw81fx238m6/container.img"`
- Longer way: Install it from source using the instructions here: https://github.com/ellisk42/ec

## Install singularity

If using openmind, singularity is already installed, so you just have to import it:
- `module add openmind/singularity`

If on Linux, install with:
- `sudo apt-get install singularity-container`
- (And then look at https://sylabs.io/guides/3.4/user-guide/installation.html if that doesn't work)

If on macOS:
- Install vagrant using brew.
- Run `vagrant up` in the root directory of this repo,
  this launches a VM according to [Vagrantfile](Vagrantfile). Customize this config file as you see fit.
- Run `vagrant ssh` from the same directory to access the VM. Singularity will be installed inside the VM. Access the git repo from inside the VM by navigating to `/vagrant`.

Make sure `singularity --version` returns a version number to make sure the installation was successful.

## Run the main file

```srun singularity exec container.img python -u bin/arc_demo.py -t 100 -g -i 5```

You should see some output enumerating solved tasks.

# Editing the files

## Editing files from scratch
To learn how to solve your own tasks from scratch, follow the tutorial from https://github.com/ellisk42/ec/blob/master/docs/creating-new-domains.md

## Editing our ARC files

For more on the ARC-specific infrastructure.
- To change the _primitives_ or _tasks_ used in a given run, edit the main file: e.g. `neurosymbolic-modules/ec/bin/arc_simon.py` (or another similarly structured file)
- To change the _implementation of the primitives_ or _implementations of tasks_, edit `neurosymbolic-modules/ec/dreamcoder/domains/arc/arcPrimitives.py` and `neurosymbolic-modules/ec/dreamcoder/domains/arc/makeTasks.py`
- To change the  _implementation of the primitives_  in OCaml, edit `neurosymbolic-modules/ec/solvers/program.ml`.  To recompile the OCaml: `cd /om2/user/$USER/neurosymbolic-modules/ec; module add openmind/singularity; ./container.img make clean; ./container.img make;`


# Resources
Here is a spreadsheet we have been used when looking at tasks to solve.
https://docs.google.com/spreadsheets/d/1qmMG2EjMMxRF4glceWPR9QgqLe-uJwRm2VUKKLqvpoE/edit?usp=sharing

In the resources, I put a couple of useful pdf files. One has all 400 ARC tasks,
for looking at different tasks. Another shows the 60/400 tasks solved in our
recent "full run".


# Common runtime/installation errors and fixes

If you get an a "dreamcoder.grammar.NoCandidates" exception, this means with the primitives you gave Dreamcoder, it can't find any programs which solve the task. Make sure the type defined for the tasks (for us, in dreamcoder/domains/arc/makeTasks.py) is what you expect, and that your primitives can be combined to give this type.

Example: if the type of the task is `arrow(tinput, tgrid)`, but your functions are all `arrow(tgrid, tgrid)` type, you need to add a primitive with type `arrow(tinput, tgrid)`.

If you get a JSON decode error when running compression, then there's probably a naming conflict with your ocaml/python primitives. For example, if you name a primitive "filter", but ocaml already had a primitive with that name used in one of the other domains. Try changing the names of your primitives to something unique, like "filter_list", or "arc_filter", and that should work.

If you get NaN's while running compression, and really weird primitives, check that your baseType("s") in python matches the make_ground "s" in ocaml. In the past we accidentally did baseType("tint") and make_ground "int".
