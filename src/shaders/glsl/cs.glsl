#version 460

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) writeonly uniform image3D img3D;

void main() {
    ivec3 coord = ivec3(gl_GlobalInvocationID.xyz);
    
    vec3 c = vec3(coord) / 256;

    imageStore(img3D, coord, vec4(c, 1.));
}