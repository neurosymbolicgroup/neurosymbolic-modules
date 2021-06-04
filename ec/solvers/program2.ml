
let primitive_get = primitive "get" (tlist(t0) @>  tint @> t0) (fun x -> x);;
(* let primitive_get_last = primitive "get_last" (tlist(t0) @> t0) (fun x -> x);; *)

let primitive_flood_fill = primitive "flood_fill" (tgrid @>  tcolor @> tgrid) (fun x -> x);;
let primitive_color = primitive "color" (tobject @> tcolor) (fun x -> x);;
let primitive_colors = primitive "colors" (tgrid @> tlist(tcolor)) (fun x -> x);;
let primitive_color_in_grid = primitive "color_in_grid" (toutput @> tcolor @> toutput) (fun x -> x);;
let primitive_group_objects_by_color = primitive "group_objects_by_color" (toriginal @> tlist(tlist(tobject)) ) (fun x -> x);;

let primitive_area = primitive "area" (tgrid @> tint) (fun x -> x);;
let primitive_move_down = primitive "move_down" (tgrid @> tgrid) (fun x -> x);;

let primitive_sortby = primitive "sortby" (tlist(t0) @>  (t0 @> t1) @> tlist(t0)) (fun x-> x);;

let primitive_draw_line = primitive "draw_line" (tgrid @> tgrid @> tdir @> tgrid) (fun x -> x);;
let primitive_draw_connecting_line = primitive "draw_connecting_line" (toriginal @> tlist(tobject) @> tgrid) (fun x -> x);;
let primitive_draw_line_slant_down = primitive "draw_line_slant_down" (toriginal @> tobject @> tgrid) (fun x -> x);;
let primitive_draw_line_slant_up = primitive "draw_line_slant_up" (toriginal @> tobject @> tgrid) (fun x -> x);;
let primitive_draw_line_down = primitive "draw_line_down" (tgrid @> tgrid) (fun x -> x);;

let primitive_objects = primitive "objects" (tgrid  @> tlist(tobject)) (fun x -> x);;

let primitive_overlay = primitive "overlay" (tgrid @> tgrid @> toutput) (fun x y -> x);;
let primitive_stack_no_crop = primitive "stack_no_crop" (tlist(tgrid) @> toutput) (fun x -> x);;

let primitive_input = primitive "input" (tinput @> toriginal) (fun x -> x);;
let primitive_object = primitive "object" (tgrid @> tgrid) (fun x -> x);;
let primitive_objects_by_color = primitive "objects_by_color" (tgrid @> tlist(tgrid)) (fun x -> x);;
let primitive_filter_list = primitive "filter_list" (tlist(t0) @> (t0 @> tboolean) @> tlist(t0)) (fun x f -> x);;
let primitive_filter_color = primitive "filter_color" (tgrid @> tcolor @> tgrid) (fun x -> x);;
let primitive_has_x_symmetry = primitive "has_x_symmetry" (tgrid @> tboolean) (fun x -> x);;
let primitive_has_y_symmetry = primitive "has_y_symmetry" (tgrid @> tboolean) (fun y -> y);;
let primitive_has_rotational_symmetry = primitive "has_rotational_symmetry" (tgrid @> tboolean) (fun x -> x);;
let primitive_rotate_ccw = primitive "rotate_ccw" (tgrid @> tgrid) (fun x -> x);;
let primitive_rotate_cw = primitive "rotate_cw" (tgrid @> tgrid) (fun x -> x);;
let primitive_combine_grids_vertically = primitive "combine_grids_vertically" (tgrid @> tgrid @> tgrid) (fun x -> x);;
let primitive_combine_grids_horizontally = primitive "combine_grids_horizontally" (tgrid @> tgrid @> tgrid) (fun x -> x);;
let primitive_x_mirror = primitive "x_mirror" (tgrid @> tgrid) (fun x -> x);;
let primitive_y_mirror = primitive "y_mirror" (tgrid @> tgrid) (fun x -> x);;
let primitive_reflect_down = primitive "reflect_down" (tgrid @> tgrid) (fun x -> x);;
let primitive_top_half = primitive "top_half" (tgrid @> tgrid) (fun x -> x);;
let primitive_bottom_half = primitive "bottom_half" (tgrid @> tgrid) (fun x -> x);;
let primitive_left_half = primitive "left_half" (tgrid @> tgrid) (fun x -> x);;
let primitive_right_half = primitive "right_half" (tgrid @> tgrid) (fun x -> x);;
