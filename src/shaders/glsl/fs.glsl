#version 450

#include <ray_marching.glsl>
#include <texture.glsl>

layout(location = 0) out vec4 f_color;

layout(push_constant) uniform AppData {
    uint time;
    vec2 screen;
    vec3 cam_position;
    vec3 cam_uu;
    vec3 cam_vv;
    vec3 cam_ww;
    Material[3] materials;
} app;

void main() {
    //f_color = texture(tex, vec3(0.0, 0.0, 0.0));
    //c_noise = tex;
    Camera camera = Camera(app.cam_position, app.cam_uu, app.cam_vv, app.cam_ww);
    vec2 coord = gl_FragCoord.xy;
    materials = app.materials;
    time = app.time;
    vec3 col = run(coord, app.screen, camera);
    f_color = vec4(col, 1.0);
}