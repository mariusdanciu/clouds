# Volumetric clouds


<img width="800" height="630" alt="image" src="https://github.com/user-attachments/assets/51659fe0-e1c9-446c-b094-808cc1953622" />


- Use Rust and Vulkano
- Generated 3D Worley noise with https://github.com/mariusdanciu/noise/tree/main. This applies periodicity for random sampling for making the texture tileable. 
- 3D texture uploaded in GPU.
- Real time rendering of volumetric clouds controlling density, isotropic + anisotropic scattering (Henyey-Greenstein phase function), sun light color.
