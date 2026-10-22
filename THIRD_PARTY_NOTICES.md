# Provenance and third-party notices

This is a fork of [jloganolson/g1-poser](https://github.com/jloganolson/g1-poser),
retaining its Git history. The inherited base commit for this preview is
`7dec2e1e6c0bfc9c5e978df5f0b300d05fe69e0b`.
No blanket replacement license is asserted for the inherited code or robot assets;
the upstream snapshot has no root LICENSE file. Consult the respective upstream
projects before reusing components outside this fork.

## Reference crawl policy and assets

The reference policy, reference deployment scene/script, crawl pose and associated
reference meshes originate from [jloganolson/g1_crawl](https://github.com/jloganolson/g1_crawl)
at commit `935e8b9c857bd494b952603c0453800a0cf5c9d2`.
That snapshot contains an MIT license attributed to Joseph Logan Olson (2025).
Its complete license text is preserved in the release asset archive at
`output/rl/release_assets/g1_crawl_LICENSE.txt`.

The policy is the upstream Git LFS artifact `sim2sim_mj/policies/policy_crawl.pt`,
299360 bytes, SHA256
`1616790c7a1f56146d8a7ed07c32e414e76a5599a0d420f3811200fd36d9b1c8`.
We do not claim to have trained this policy. The release adds a controller adapter,
ankle target restriction and bounded task-space lift correction.

The modified runtime model derives from this fork's G1 model; the printed forearm
collision hull derives from the reference project's `3d-printed-parts` geometry.
These collision approximations and the contact solver profile are not hardware
calibrations. Robot and mesh provenance is retained rather than assigning a new
license to third-party geometry.

The release archive has a per-file SHA256 manifest, and the installer pins the
whole archive hash. The upstream deployment script is included as provenance and
parsed only for literal constants; its GUI, device-control and gamepad code are
not executed by the crawl runtime.

MuJoCo, PyTorch, NumPy, MINK and other dependencies retain their own licenses;
they are installed through the package manager, not vendored as source here.
