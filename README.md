# Volumetric clouds


<img width="804" height="635" alt="image" src="https://github.com/user-attachments/assets/ad3be816-271b-4c68-a00a-364397016dbb" />



- Use Rust and Vulkano
- Generated 3D Worley noise with https://github.com/mariusdanciu/noise/tree/main. This applies periodicity for random sampling for making the texture tileable. 
- 3D texture uploaded in GPU.
- Real time rendering of volumetric clouds controlling density, isotropic + anisotropic scattering (Henyey-Greenstein phase function), sun light color.
