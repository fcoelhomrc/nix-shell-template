# nix-shell-template

Reproducible dev shells via Nix flakes. Includes CUDA native libs for Python ML workflows (torch, numpy, etc.).

## Usage

### Copy flake into new project
```sh 
nix flake init -t github:fcoelhomrc/nix-shell-template
```

### With direnv (recommended)

```sh
direnv allow   # first time only — re-runs automatically on flake.lock changes
```

### Without direnv

```sh
nix develop            # default shell (Python + CUDA)
nix develop .#minimal  # Python only, no CUDA
```

## Adding packages

Edit `flake.nix` → `packages = [ pkgs.uv pkgs.whatever ];`

Run `nix flake update` to bump nixpkgs.

## Finding missing native libs

```sh
find .venv/ -name "*.so" | xargs ldd | grep "not found" | sort | uniq
```

Add missing libs to `NIX_LD_LIBRARY_PATH` in `flake.nix`.
