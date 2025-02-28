#include "cow.cpp"
int main() {
    #define RING_BUFFER_LENGTH 128
    vec2 ring_buffer[RING_BUFFER_LENGTH] = {};
    vec2 *ring_buffer_write_head = ring_buffer;
    vec2 *ring_buffer_one_past_end = ring_buffer + RING_BUFFER_LENGTH;

    // glDisable(GL_CULL_FACE);

    Camera camera = make_Camera2D(256.0f, {}); // the screen is 256.0f World units tall
    while (begin_frame(&camera)) {
        mat4 PV = camera.get_PV();
        vec2 mouse_position = get_mouse_position_World(PV);

        eso_begin(NDC_from_Pixel, SOUP_TRIANGLES); // Pixel coordinates
        eso_color(basic.blue);
        eso_vertex(0.0f, 0.0f);
        eso_vertex(0.0f, 100.0f);
        eso_vertex(100.0f, 0.0f);
        eso_end();

        eso_begin(PV, SOUP_TRIANGLES); // World coordinates
        eso_color((mouse_left_held) ? basic.green : basic.gray);
        eso_vertex(mouse_position);
        eso_vertex(mouse_position + V2(100.0f, 0));
        eso_vertex(mouse_position + V2(0, 100.0f));
        eso_end();

        eso_begin(PV, SOUP_POINTS); // World coordinates
        eso_overlay(true);
        eso_color(monokai.blue);
        eso_size(64.0f);
        eso_vertex(0.0f, 0.0f);
        eso_end();

        // key_pressed, key_held, key_released
        if (!key_toggled['P']) {
            *(ring_buffer_write_head++) = mouse_position;
            if (ring_buffer_write_head == ring_buffer_one_past_end) ring_buffer_write_head = ring_buffer;
        }

        eso_begin(PV, SOUP_LINE_STRIP);
        eso_color(basic.red);
        eso_size(5.0f);
        real t = 0.0f;
        real dt = (1.0f / (RING_BUFFER_LENGTH - 1));
        for (vec2 *point = ring_buffer_write_head; point != ring_buffer_one_past_end; ++point) {
            eso_color(color_rainbow_swirl(t));
            t += dt;
            eso_vertex(*point);
        }
        for (vec2 *point = ring_buffer; point != ring_buffer_write_head; ++point) {
            eso_color(color_rainbow_swirl(t));
            t += dt;
            eso_vertex(*point);
        }
        eso_end();
    }
    return 1;
}
