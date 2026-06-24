#version 450
#include <vulkano.glsl>

layout(location = 0) in vec2 v_tex_coords;
layout(location = 0) out vec4 f_color;

layout(push_constant) uniform PushConstantData {
    SamplerId sampler_id;
    SampledImageId dst_image;
};

vec4 samp(vec2 uv) {
    return texture(vko_sampler2D(dst_image, sampler_id), uv);
}

void main() {
    vec4 color = samp(v_tex_coords);
    f_color = vec4(color.rgb, 1.0);
}
