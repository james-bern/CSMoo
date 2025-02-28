```c
mat4 NDC_from_Pixel;

bool mouse_left_pressed;
bool mouse_left_held;
bool mouse_left_released;

bool mouse_right_pressed;
bool mouse_right_held;
bool mouse_right_released;

vec2 mouse_position_Pixel;
vec2 mouse_position_NDC;
vec2 get_mouse_position_World(mat4 PV);
vec2 _get_mouse_change_in_position_World(mat4 PV);

bool key_pressed[];
bool key_held[];
bool key_released[];
bool key_toggled[];
```
