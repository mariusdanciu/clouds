#include <common.glsl>
#include <texture.glsl>

Hit sdf(vec3 p) {

    vec3 q = p - vec3(-0.5, 0.8, 0.0);
    float d = box_sdf(q, vec3(4.5, 1., 4.5), 0.1);

    int material = 0;

    vec3 col = materials[material].color;

    //c_noise = tex;    
    return Hit(d, 1, col, 0.0, true);
}