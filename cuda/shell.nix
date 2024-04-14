with import <nixpkgs> { };
stdenv.mkDerivation {
  name = "env";
  nativeBuildInputs = [ pkg-config ];
  buildInputs = [
    opencv4
    cudatoolkit_11
    # cuda_nvcc
    # nccl
  ];
}
