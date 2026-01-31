# Volumetric clouds


<img width="808" height="640" alt="image" src="https://github.com/user-attachments/assets/ee57cf2b-61bd-4f35-95b4-61151d3815a3" />



- Use Rust and Vulkano
- Generated 3D Worley noise with https://github.com/mariusdanciu/noise/tree/main. This applies periodicity for random sampling for making the texture tileable. 
- 3D texture uploaded in GPU.
- Real time rendering of volumetric clouds controlling density, isotropic + anisotropic scattering (Henyey-Greenstein phase function), sun light color.
