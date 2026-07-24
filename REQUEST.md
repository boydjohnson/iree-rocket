# Summary

the current (iree-rocket) will be the new "mono-repo" repository that will contain the work in


boydjohnson@laplace:~/projects$ ls -l | grep -E 'iree|rocket'
drwxrwxr-x  6 boydjohnson boydjohnson     4096 Jul 19 19:45 iree-build
drwxrwxr-x  6 boydjohnson boydjohnson     4096 Jul 24 12:08 iree-rocket-hal
drwxrwxr-x  5 boydjohnson boydjohnson     4096 Jul 24 07:05 rocket-compiler-plugin
drwxrwxr-x  7 boydjohnson boydjohnson     4096 Jul 24 14:59 rocket-hal-driver
drwxrwxr-x 10 boydjohnson boydjohnson     4096 Jul 23 17:19 rocket-schema

## artifacts that this repository will produce

The repository should be able to build `iree-compile` and `iree-run-module` and `iree-benchmark-module`


## request

Can you plan a mono repo design?
