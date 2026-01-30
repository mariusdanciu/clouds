#include <scene.glsl>
#include <texture.glsl>

#define MAX_STEPS 180
#define MAX_VOLUME_STEPS 280

#define HIT_PRECISION 0.001
#define MAX_DISTANCE 100.0

float CLOUD_SCALE = 0.2;
vec3 CLOUD_OFFSET = vec3(0);
float DENSITY_THRESHOLD = 0.2;
float DENSITY_MULTIPLIER = 0.2;
float LIGHT_STEP_SIZE = 0.3;
float VOLUME_STEP_SIZE = 0.06;
float ABSORPTION_COEFF = 1.0;

vec3 SUN_COLOR = vec3(1.0, 0.85, 0.7);

vec3 normal(vec3 p) {
    float k = 0.5773 * 0.0005;
    vec2 e = vec2(1., -1.);

    vec3 xyy = vec3(e.x, e.y, e.y);
    vec3 yyx = vec3(e.y, e.y, e.x);
    vec3 yxy = vec3(e.y, e.x, e.y);
    vec3 xxx = vec3(e.x, e.x, e.x);

    vec3 r_xyy = p + xyy * k;
    vec3 r_yyx = p + yyx * k;
    vec3 r_yxy = p + yxy * k;
    vec3 r_xxx = p + xxx * k;

    return normalize(xyy * sdf(r_xyy).dist + yyx * sdf(r_yyx).dist + yxy * sdf(r_yxy).dist + xxx * sdf(r_xxx).dist);
}

float occlusion(vec3 pos, vec3 nor) {
    float occ = 0.0;
    float sca = 1.0;
    for(int i = 0; i < 5; i++) {
        float hr = 0.02 + 0.025 * (i * i);

        Hit hit = sdf(pos + nor * hr);

        occ += -(hit.dist - hr) * sca;
        sca *= 0.85;
    }
    return 1.0 - clamp(occ, 0.0, 1.0);
}

float shadow(Ray ray, float k) {
    float res = 1.0;

    float t = 0.01;

    for(int i = 0; i < 64; i++) {
        vec3 pos = ray.origin + ray.direction * t;
        float h = sdf(pos).dist;

        res = min(res, k * (max(h, 0.0) / t));
        if(res < 0.0001) {
            break;
        }
        t += clamp(h, 0.01, 5.0);
    }

    return res;
}

float sample_density(vec3 pos) {
    float noise = texture(tex, pos * CLOUD_SCALE).r;

    float density = max(0.0, noise - DENSITY_THRESHOLD) * DENSITY_MULTIPLIER;
    return density;
}

float light_march(vec3 pos, vec3 light_dir) {
    float t = 0.0;
    float total_density = 0.0;

    for(int i = 0; i < 6; i++) {
        vec3 lightPos = pos + light_dir * t;
        float density = sample_density(lightPos);

        total_density += density * LIGHT_STEP_SIZE;
        t += LIGHT_STEP_SIZE;

    }
    float transmittance = exp(-total_density * ABSORPTION_COEFF);
    return transmittance;
}

Hit ray_march(Ray ray, vec3 sky) {
    float t = 0.0;

    float ambient_strength = 0.7;
    float transmittance = 1.0;
    vec3 accumulatedColor = vec3(0.0);
    vec3 light_dir = -normalize(vec3(-3., -1.5, -2.));

    float step = 0;
    bool inside_volume = false;

    float total_density = 0.0;
    float distance_traveled = 0.0;
    float light_energy = 0.0;

    for(int i = 0; i < MAX_STEPS; i++) {
        if(t > MAX_DISTANCE) {
            break;
        }

        vec3 pos = ray.origin + ray.direction * t;
        Hit h = sdf(pos);

        // if(h.dist < 0.001) {
        //     // Touched the volume

        //     float density = sample_density(pos);
        //     vec4 col = texture(tex, pos * 0.1);

        //     return Hit(t, h.material_index, col.xxx, true);
        // }

        if(h.dist <= 0.0) {
            // Inside the volume
            step = VOLUME_STEP_SIZE;
            inside_volume = true;
            float density = sample_density(pos);

            if(density > 0.0) {
                total_density += density * VOLUME_STEP_SIZE;
                distance_traveled += VOLUME_STEP_SIZE;

                float light_transmittance = light_march(pos, light_dir);

                // float light_energy = light_march(pos, light_dir);

                // // Henyey-Greenstein phase function (optional, for anisotropic scattering)
                float cosTheta = dot(ray.direction, light_dir);
                float g = 0.8; // forward scattering parameter
                float forward = (1.0 - g * g) / (4.0 * 3.14159 * pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5));

                float isotropic = 1.0 / (4. * 3.141592) * 10.0;
                float phase = mix(isotropic, forward, 0.3);

                light_energy += density * transmittance * light_transmittance * VOLUME_STEP_SIZE * phase;
                transmittance *= exp(-density * ABSORPTION_COEFF);

                accumulatedColor += light_energy * transmittance * SUN_COLOR;

                // vec3 lightning = SUN_COLOR * phase * light_energy * 5;
                // float multiple_scatter = pow(density, 0.5) * 0.1;
                // vec3 ambient = sky * multiple_scatter;

                // lightning += ambient;

                // float scattering_coeff = 1.0;
                // float extintion_coeff = ABSORPTION_COEFF;

                // accumulatedColor += transmittance * density * lightning * VOLUME_STEP_SIZE * scattering_coeff;

                // float sample_transmittance = exp(-density * extintion_coeff * VOLUME_STEP_SIZE);
                // transmittance *= sample_transmittance;

                if(transmittance < 0.01)
                    break;
            }

        } else {
            // if(inside_volume) {
            //     transmittance = exp(-total_density);
            // }
            inside_volume = false;
            step = max(h.dist * 0.5, 0.01);
        }

        t += step;

    }
    return Hit(t, 0, accumulatedColor, transmittance, false);
}

vec3 fog2(
    in vec3 col,   // color of pixel
    in float t,     // distance to point
    in vec3 rd,    // camera to point
    in vec3 lig
)  // sun direction
{
    float fogAmount = 1.0 - exp(-t * 0.05);
    float sunAmount = max(dot(rd, lig), 0.0);
    vec3 fogColor = mix(vec3(0.5, 0.6, 0.7), // blue
    vec3(1.0, 0.9, 0.7), // yellow
    pow(sunAmount, 10.0));
    return mix(col, fogColor, fogAmount);
}

vec3 path_trace(Ray ray, DirectionalLight d_light, vec3 res, vec3 sky, int bounce) {

    vec3 refl_col = vec3(0);
    float refl_roughness = -1.0;

    Hit hit;

    for(int bounce = 0; bounce < 1; bounce++) {

        hit = ray_march(ray, sky);

        if(hit.hit) {
            vec3 p = ray.origin + ray.direction * hit.dist;
            vec3 n = normal(p);
            vec3 light_dir = -d_light.direction;
            float occlusion = occlusion(p, n);
            float shadow = 1.0; //shadow(Ray(p + n * 0.0001, light_dir), 32);

            Material material = materials[hit.material_index];

            float mat_specular = material.specular;
            float mat_shininess = material.shininess;

            vec3 col = hit.color;

            vec3 half_angle = normalize(-ray.direction + light_dir);

            float shininess = pow(max(dot(n, half_angle), 0.), mat_shininess);

            float sun = clamp(dot(n, light_dir), 0.0, 1.0);
            float indirect = 0.1 * clamp(dot(n, normalize(light_dir * vec3(-1.0, 0.0, -1.0))), 0.0, 1.0);

            vec3 light = material.diffuse * sun * d_light.color * pow(vec3(shadow), vec3(1.3, 1.2, 1.5));

            light += sky * vec3(0.16, 0.20, 0.28) * occlusion;
            light += indirect * vec3(0.40, 0.28, 0.20) * occlusion;
            light += mat_specular * shininess * shadow;

            col *= light * d_light.intensity;

            //col = fog2(col, hit.dist, ray.direction, light_dir);
            res = clamp(col, 0.0, 1.0);

            if(refl_roughness >= 0) {
                res = mix(res, refl_col, refl_roughness);
            }

            if(material.roughness < 1.0) {
                vec3 refl = normalize(reflect(ray.direction, n));
                ray = Ray(p + n * 0.01, refl);
                refl_col = res;
                refl_roughness = material.roughness;
            } else {
                refl_roughness = -1;
            }

        } else {
            if(refl_roughness >= 0) {
                res = mix(res, refl_col, refl_roughness);
            }

            res = hit.color + res * hit.transmittance;

            break;
        }
    }
    return res;
}

vec3 run(vec2 coord, vec2 screen, Camera camera) {
    vec2 uv = (2 * coord - screen) / screen.y;
    uv.y = -uv.y;

    Ray ray = Ray(camera.position, normalize(uv.x * camera.uu + uv.y * camera.vv + 1.5 * camera.ww));
    DirectionalLight d_light = DirectionalLight(normalize(vec3(-3., -1.5, -2.)), SUN_COLOR, 1.0);

    vec3 sky = clamp(vec3(0.5, 0.8, 1.) - (0.7 * ray.direction.y), 0.0, 1.0);

    sky = mix(sky, vec3(0.5, 0.7, 0.9), exp(-10.0 * max(ray.direction.y, 0.0)));
    //sky = vec3(0.);
    vec3 res = sky;

    float sundot = clamp(dot(ray.direction, -d_light.direction), 0.0, 1.0);

    res += 0.25 * vec3(1.0, 0.7, 0.4) * pow(sundot, 5.0);
    res += 0.25 * vec3(1.0, 0.6, 0.6) * pow(sundot, 64.0);
    res += 0.25 * vec3(1.0, 0.9, 0.6) * pow(sundot, 512.0);

    res = path_trace(ray, d_light, res, sky, 0);

    res = pow(res, vec3(0.4545));
    return res;
}